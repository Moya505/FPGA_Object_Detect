# model_vitis_compatible.py - YOLOv11 model optimized for Vitis AI compatibility
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def autopad(k, p=None, d=1):
    """Auto-padding calculation"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    """Depthwise convolution"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, d=d, g=math.gcd(c1, c2), act=act)

class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(nn.Module):
    """C3k module"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2(nn.Module):
    """C3k2 module"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        if c3k:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# SIMPLIFIED ATTENTION MECHANISM - More compatible with Vitis AI
class SimpleAttention(nn.Module):
    """Simplified attention mechanism for Vitis AI compatibility"""
    def __init__(self, dim):
        super().__init__()
        # Use simpler operations that are more FPGA-friendly
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv(dim, dim // 4, 1)
        self.fc2 = Conv(dim // 4, dim, 1, act=False)

    def forward(self, x):
        # Channel attention using global average pooling
        b, c, h, w = x.shape
        
        # Global average pooling
        y = self.global_pool(x)  # [B, C, 1, 1]
        
        # Channel attention
        y = self.fc1(y)  # [B, C//4, 1, 1]
        y = self.fc2(y)  # [B, C, 1, 1]
        
        # Apply attention using element-wise multiplication
        # This is simpler than multi-head attention and more FPGA-friendly
        y = torch.sigmoid(y)
        
        return x * y

class SimplePSABlock(nn.Module):
    """Simplified PSA Block for Vitis AI compatibility"""
    def __init__(self, c, shortcut=True):
        super().__init__()
        # Replace complex attention with simpler channel attention
        self.attn = SimpleAttention(c)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        # Simplified attention path
        if self.add:
            x = x + self.attn(x)
            x = x + self.ffn(x)
        else:
            x = self.attn(x)
            x = self.ffn(x)
        return x

class SimpleC2PSA(nn.Module):
    """Simplified C2PSA module for Vitis AI compatibility"""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        # Use simplified PSA blocks
        self.m = nn.Sequential(*(SimplePSABlock(self.c) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DFL(nn.Module):
    """Distribution Focal Loss (DFL)"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class Detect(nn.Module):
    """YOLOv11 Detect head for Vitis AI"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Forward pass - return raw outputs for Vitis AI compatibility"""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x

class YOLOv11VitisCompatible(nn.Module):
    """
    YOLOv11 model optimized for Vitis AI compatibility
    
    Changes made:
    1. Replaced complex attention with simplified channel attention
    2. Removed problematic elementwise operations
    3. Simplified PSA blocks
    4. Maintained same overall architecture
    """
    def __init__(self, nc=11):
        super().__init__()
        
        # Backbone (Layers 0-10) - same as before but with simplified attention
        self.backbone = nn.ModuleList([
            Conv(3, 16, 3, 2),                    # 0: 3->16
            Conv(16, 32, 3, 2),                   # 1: 16->32
            C3k2(32, 64, 1, False),               # 2: 32->64
            Conv(64, 64, 3, 2),                   # 3: 64->64
            C3k2(64, 128, 1, False),              # 4: 64->128 (P3)
            Conv(128, 128, 3, 2),                 # 5: 128->128
            C3k2(128, 128, 1, True),              # 6: 128->128 (P4)
            Conv(128, 256, 3, 2),                 # 7: 128->256
            C3k2(256, 256, 1, True),              # 8: 256->256
            SPPF(256, 256, 5),                    # 9: 256->256
            SimpleC2PSA(256, 256, 1)              # 10: 256->256 (P5) - SIMPLIFIED!
        ])
        
        # Neck/FPN (Layers 11-22) - same as before
        self.neck = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 11
            Concat(),                                      # 12
            C3k2(384, 128, 1, False),                     # 13: 384->128
            nn.Upsample(scale_factor=2, mode='nearest'),  # 14
            Concat(),                                      # 15
            C3k2(256, 64, 1, False),                      # 16: 256->64
            Conv(64, 64, 3, 2),                           # 17: 64->64
            Concat(),                                      # 18
            C3k2(192, 128, 1, False),                     # 19: 192->128
            Conv(128, 128, 3, 2),                         # 20: 128->128
            Concat(),                                      # 21
            C3k2(384, 256, 1, True),                      # 22: 384->256
        ])
        
        # Detection head (Layer 23)
        self.detect = Detect(nc, (64, 128, 256))
        self.detect.stride = torch.tensor([8., 16., 32.])

    def forward(self, x):
        """Forward pass - ONLY forward method for Vitis AI compatibility"""
        # Backbone forward pass (layers 0-10)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 4:    # After layer 4: 128 channels (P3)
                p3 = x
            elif i == 6:  # After layer 6: 128 channels (P4)  
                p4 = x
            elif i == 10: # After layer 10: 256 channels (P5)
                p5 = x
        
        # Neck/FPN forward pass (layers 11-22)
        x = self.neck[0](p5)              # Layer 11: Upsample P5
        x = self.neck[1]([x, p4])         # Layer 12: Concat
        x = self.neck[2](x)               # Layer 13: C3k2
        n4 = x
        
        x = self.neck[3](x)               # Layer 14: Upsample
        x = self.neck[4]([x, p3])         # Layer 15: Concat
        x = self.neck[5](x)               # Layer 16: C3k2
        n3 = x
        
        x = self.neck[6](n3)              # Layer 17: Conv
        x = self.neck[7]([x, n4])         # Layer 18: Concat
        x = self.neck[8](x)               # Layer 19: C3k2
        n4_out = x
        
        x = self.neck[9](x)               # Layer 20: Conv
        x = self.neck[10]([x, p5])        # Layer 21: Concat
        x = self.neck[11](x)              # Layer 22: C3k2
        n5_out = x
        
        # Detection head (layer 23)
        detections = self.detect([n3, n4_out, n5_out])
        
        return detections

# Alias for compatibility
YOLOv11 = YOLOv11VitisCompatible

if __name__ == "__main__":
    # Test the simplified model
    model = YOLOv11VitisCompatible(nc=11)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Vitis AI compatible model created!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of outputs: {len(output)}")
    for i, out in enumerate(output):
        print(f"Output {i} shape: {out.shape}")
    
    # Test JIT tracing
    try:
        traced = torch.jit.trace(model, dummy_input)
        print("✓ JIT tracing successful!")
    except Exception as e:
        print(f"✗ JIT tracing failed: {e}")