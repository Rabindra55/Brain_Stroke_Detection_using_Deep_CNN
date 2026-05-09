import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

#model

def conv(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

def up_conv(in_channels,out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2),
                         nn.ReLU(inplace=True))
    
from torchvision.models import vgg16_bn

class Unet(nn.Module):
    def __init__(self,pretrained=True,out_channels=3):
        super(Unet,self).__init__()

        self.encoder=vgg16_bn(pretrained=pretrained).features
        self.block1=nn.Sequential(*self.encoder[:6])
        self.block2=nn.Sequential(*self.encoder[6:13])
        self.block3=nn.Sequential(*self.encoder[13:20])
        self.block4=nn.Sequential(*self.encoder[20:27])
        self.block5=nn.Sequential(*self.encoder[27:34])

        self.bottleneck=nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck=conv(512,1024)
        
        self.up_conv6=up_conv(1024,512)
        self.conv6=conv(512+512,512)
        self.up_conv7=up_conv(512,256)
        self.conv7=conv(256+512,256)
        self.up_conv8=up_conv(256,128)
        self.conv8=conv(128+256,128)
        self.up_conv9=up_conv(128,64)
        self.conv9=conv(64+128,64)
        self.up_conv10=up_conv(64,32)
        self.conv10=conv(32+64,32)
        self.conv11=nn.Conv2d(32,out_channels,kernel_size=1)

    def forward(self,x):
        block1=self.block1(x)
        block2=self.block2(block1)
        block3=self.block3(block2)
        block4=self.block4(block3)
        block5=self.block5(block4)
        
        bottleneck=self.bottleneck(block5)
        x=self.conv_bottleneck(bottleneck)

        x=self.up_conv6(x)
        x=torch.cat([x,block5],dim=1)
        x=self.conv6(x)

        x=self.up_conv7(x)
        x=torch.cat([x,block4],dim=1)
        x=self.conv7(x)

        x=self.up_conv8(x)
        x=torch.cat([x,block3],dim=1)
        x=self.conv8(x)

        x=self.up_conv9(x)
        x=torch.cat([x,block2],dim=1)
        x=self.conv9(x)

        x=self.up_conv10(x)
        x=torch.cat([x,block1],dim=1)
        x=self.conv10(x)

        x=self.conv11(x)

        return x

#################################################################
# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cpu"

CLASS_NAMES = ["Background", "Bleeding", "Hemorrhage"]

# Colors (BGR for OpenCV)
COLORS = {0:(0,0,0),
    1: (255, 0, 255),    # Bleeding → Red
    2: (255,78,89)   # Hemorrhage → Yellow
}

st.set_page_config(
    page_title="AI Brain Stroke Segmentation System",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model=Unet()
    model.load_state_dict(torch.load('C:/Users/rg528/Downloads/brain_stroke_detection_using_deep_cnn/brain_stroke_detection/brain_stroke_model.pth',map_location='cpu'))
    model.eval()
    COLORS ={0:(0,0,0),1:(255,0,0),2:(255,78,89)}

    return model

model = load_model()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Visualization Controls")

opacity = st.sidebar.slider("Overlay Opacity", 0.1, 1.0, 0.5)

# -----------------------------
# PREPROCESS
# -----------------------------

def preprocess(image):
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

# -----------------------------
# POSTPROCESS (MULTI-CLASS)
# -----------------------------
def get_multiclass_mask(output):
    """
    Output shape: [1, C, H, W]
    """
    probs = torch.softmax(output, dim=1)
    mask = torch.argmax(probs, dim=1)
    return mask.squeeze().cpu().numpy()

# -----------------------------
# OVERLAY
# -----------------------------
def create_overlay(image, mask, opacity):
    image = np.array(image.resize((224, 224)))
    overlay = image.copy()

    for cls, color in COLORS.items():
        overlay[mask == cls] = color

    return cv2.addWeighted(image, 1 - opacity, overlay, opacity, 0)

# -----------------------------
# HEADER
# -----------------------------
st.title("🧠 AI Brain Stroke Segmentation")
st.markdown("### Multi-Class Segmentation (Bleeding vs Hemorrhage)")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CT/MRI Scan", type=["png", "jpg", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🖼 Original Scan")
        st.image(image, use_container_width=True)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    input_tensor = preprocess(image)

    with torch.no_grad():
        output = model(input_tensor)
        mask = get_multiclass_mask(output)

    overlay = create_overlay(image, mask, opacity)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    with col2:
        st.subheader("🧬 Segmentation Mask")
        st.image(mask.astype(np.uint8) * 120, use_container_width=True)

    with col3:
        st.subheader("🔥 Overlay")
        st.image(overlay, use_container_width=True)

    # -----------------------------
    # ANALYSIS
    # -----------------------------
    st.markdown("---")
    st.subheader("📊 Region Analysis")

    total_pixels = mask.size
    bleeding_pixels = np.sum(mask == 1)
    hemorrhage_pixels = np.sum(mask == 2)

    bleeding_pct = (bleeding_pixels / total_pixels) * 100
    hemorrhage_pct = (hemorrhage_pixels / total_pixels) * 100

    c1, c2, c3 = st.columns(3)

    c1.metric("Bleeding %", f"{bleeding_pct:.2f}%")
    c2.metric("Hemorrhage %", f"{hemorrhage_pct:.2f}%")
    c3.metric("Total Affected %", f"{bleeding_pct + hemorrhage_pct:.2f}%")

    # -----------------------------
    # BAR CHART
    # -----------------------------
    st.subheader("📈 Distribution")
    chart_data = {
        "Bleeding": bleeding_pct,
        "Hemorrhage": hemorrhage_pct
    }
    st.bar_chart(chart_data)

    # -----------------------------
    # DIAGNOSIS LOGIC
    # -----------------------------
    st.subheader("🧾 Diagnosis")

    if bleeding_pct ==0 and hemorrhage_pct ==0:
        st.success("🟢 No significant abnormality detected")
    elif bleeding_pct > hemorrhage_pct:
        st.warning("🟠 Predominant Bleeding Detected")
    else:
        st.error("🔴 Hemorrhage Dominant — Critical Condition")

else:
    st.info("Upload a scan to begin analysis")

