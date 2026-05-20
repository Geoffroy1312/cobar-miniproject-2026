import numpy as np
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


class FlyVisionModel:
    def __init__(self):
        self.model = FlyUNet()
        # Charger les poids pré-entraînés
        self.model.load_state_dict(torch.load("submission/fly_unet_best.pth", map_location=torch.device('cpu')))
        self.model.eval()  # Mettre le modèle en mode évaluation

    def get_1d_occupancy_grid(self, image, kernel_size=(12, 12), safety_margin=51):
        """
        Génère une grille d'occupation 1D à partir de la vision.
        """
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            isolated = self.model(image_tensor)
            
        mask = (isolated.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)
        
        if safety_margin > 0:
            safety_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (safety_margin, safety_margin))
            cleaned_mask = cv2.dilate(cleaned_mask, safety_kernel, iterations=1)
        
        occupancy_1d = (np.max(cleaned_mask, axis=0) > 0).astype(np.uint8)
        
        return occupancy_1d

    def get_debug_masks(self, image, kernel_size=(15, 15), safety_margin=51):
        """
        Renvoie les masques binaires intermédiaires pour la visualisation et le débuggage.
        
        Returns:
            raw_mask: Sortie brute du réseau U-Net (binaire 0 ou 255).
            cleaned_mask: Masque après l'ouverture morphologique (suppression du bruit).
            final_mask: Masque après la dilatation (marge de sécurité).
        """
        # 1. Inférence brute
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            isolated = self.model(image_tensor)
        raw_mask = (isolated.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        
        # 2. Nettoyage (Ouverture morphologique)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, clean_kernel)
        
        # 3. Marge de sécurité (Dilatation)
        if safety_margin > 0:
            safety_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (safety_margin, safety_margin))
            final_mask = cv2.dilate(cleaned_mask, safety_kernel, iterations=1)
        else:
            final_mask = cleaned_mask.copy()
            
        return raw_mask, cleaned_mask, final_mask