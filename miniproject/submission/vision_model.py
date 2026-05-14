import numpy as np
from torch import dist
import cv2




import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 2. ARCHITECTURE DU RÉSEAU (U-NET XL & ROBUSTE)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class FlyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODEUR (Descente) ---
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # 5ème niveau ajouté
        self.down5 = DoubleConv(256, 512)
        self.pool5 = nn.MaxPool2d(2)

        # --- BOTTLENECK (Goulot d'étranglement) ---
        # Capacité massive pour retenir le contexte global
        self.bottleneck = DoubleConv(512, 1024)

        # --- DECODEUR (Montée) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) # 512 (up1) + 512 (down5) = 1024

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)  # 256 (up2) + 256 (down4) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)  # 128 (up3) + 128 (down3) = 256

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)   # 64 (up4) + 64 (down2) = 128

        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv5 = DoubleConv(64, 32)    # 32 (up5) + 32 (down1) = 64

        # --- SORTIE ---
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # --- Descente ---
        x1 = self.down1(x)
        p1 = self.pool1(x1)

        x2 = self.down2(p1)
        p2 = self.pool2(x2)

        x3 = self.down3(p2)
        p3 = self.pool3(x3)

        x4 = self.down4(p3)
        p4 = self.pool4(x4)

        x5 = self.down5(p4)
        p5 = self.pool5(x5)

        # --- Bottleneck ---
        bn = self.bottleneck(p5)

        # --- Montée (avec Skip Connections sécurisées) ---
        u1 = self.up1(bn)
        if u1.shape != x5.shape:
            u1 = F.interpolate(u1, size=x5.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, x5], dim=1)
        u1 = self.up_conv1(u1)

        u2 = self.up2(u1)
        if u2.shape != x4.shape:
            u2 = F.interpolate(u2, size=x4.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, x4], dim=1)
        u2 = self.up_conv2(u2)

        u3 = self.up3(u2)
        if u3.shape != x3.shape:
            u3 = F.interpolate(u3, size=x3.shape[2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up_conv3(u3)

        u4 = self.up4(u3)
        if u4.shape != x2.shape:
            u4 = F.interpolate(u4, size=x2.shape[2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, x2], dim=1)
        u4 = self.up_conv4(u4)

        u5 = self.up5(u4)
        if u5.shape != x1.shape:
            u5 = F.interpolate(u5, size=x1.shape[2:], mode='bilinear', align_corners=False)
        u5 = torch.cat([u5, x1], dim=1)
        u5 = self.up_conv5(u5)

        return self.out_conv(u5)


class FlyVisionModel :
    def __init__(self):
        self.model = FlyUNet()
        # Charger les poids pré-entraînés (assurez-vous que le chemin est correct)
        self.model.load_state_dict(torch.load("submission/fly_unet_best.pth", map_location=torch.device('cpu')))
        self.model.eval()  # Mettre le modèle en mode évaluation

    def detect_grass(self,image):
        """
        Détecte les triangles sur une image concaténée (gauche + droite).
        La distance retournée est négative si le triangle est à gauche, positive s'il est à droite.
        """
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Convertir en tensor et normaliser
        isolated = self.model(image_tensor)  # Passer l'image à travers le modèle pour obtenir la segmentation
        isolated = (isolated.squeeze().detach().numpy() > 0.5).astype(np.uint8) * 255  # Seuil pour obtenir une image binaire
        # Détecte les contours
        contours, _ = cv2.findContours(isolated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Centre de l'image globale
        center_image_x = image.shape[1] / 2.0
        
        
        big_contours = []
        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 2 and cv2.arcLength(approx, True) > 15:
                contour_center_x = np.mean(approx[:, 0, 0])
                distance = (contour_center_x - center_image_x)
                size = (approx[:, 0, 1].max() - approx[:, 0, 1].min())
                if abs(distance) < 20 :
                    distance = abs(distance)
                    
                if distance == 0 :
                    #avoid division by zero 
                    score = np.inf
                else :
                    score = size / (abs(distance)*0.5)

                big_contours.append([approx , distance , size, score])

        
        return big_contours