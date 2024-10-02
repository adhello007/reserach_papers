import torch 
import torch.nn as nn 
import torch.nn.functional as F

'''
    THIS IS A STRICT RESERACH PAPER IMPLEMENTATION WHERE THERE ARE UNPADDED CONVOLUTIONS, A NEED FOR CENTER CROPPING, 
    AND NO BATCH NORMALIZATION 
''' 



class DoubleConv(nn.Module):
    def __init__(self, in_channels: int,out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3), #unpadded convolutions 
            nn.ReLU(inplace=True), #inplace=True helps save memory as it makes changes to the memory locations instead of making a copy. 
            nn.Conv2d(out_channels, out_channels, kernel_size=3), #unpadded convolutions 
            #the output of the second convolution is the same depth as the output of the first.
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
    


class Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        '''
            Here we are building the exact architecture blocks. 
            Then we will put it together in forward()
        ''' 
    
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        #define bottleneck (1024)
        self.bottleneck = DoubleConv(512,1024)

        self.upconv4 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2) #upconv needs a stride of 2, kern of 2
        self.dec4 = DoubleConv(1024, 512) #here the input is still 1024 because, we concatenate the skipped connections at the first layer of every major double conv block 
        self.upconv3 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.dec3 = DoubleConv(512,256) 
        self.upconv2 = nn.ConvTranspose2d(256,128, kernel_size=2, stride =2)
        self.dec2 = DoubleConv(256,128) 
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128,64)


        #final output layer of 1by 1 conv kernel
        self.final_conv = nn.Conv2d(64,out_channels, kernel_size=1)

    def center_crop(self,enc_features: torch.Tensor, dec_features: torch.Tensor) -> torch.Tensor:
        """ Center crop the encoder feature to match the size of the decoder feature """
        
        _,_,enc_h, enc_w  = enc_features.size()
        _,_,dec_h, dec_w = dec_features.size()
    
        crop_top = (enc_h - dec_h) // 2
        crop_left = (enc_w - dec_w) // 2 

        return enc_features[:,:, crop_top:crop_top + dec_h, crop_left:crop_left+dec_w]


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1,kernel_size=2))  
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))      
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        #bottleneck 
        bottleneck = self.bottleneck(F.max_pool2d(enc4,2))

        #decoding path with skip connections 
        upconv4 = self.upconv4(bottleneck)
        skip_con4_add = torch.cat((self.center_crop(enc4, upconv4), upconv4), dim=1)
        dec4 = self.dec4(skip_con4_add)

        upconv3 = self.upconv3(dec4)
        skip_con3_add = torch.cat((self.center_crop(enc3,upconv3), upconv3), dim=1) 
        dec3 = self.dec3(skip_con3_add)

        upconv2 = self.upconv2(dec3)
        skip_con2_add = torch.cat((self.center_crop(enc2,upconv2), upconv2), dim=1) 
        dec2 = self.dec2(skip_con2_add)

        upconv1 = self.upconv1(dec2)
        skip_con1_add = torch.cat((self.center_crop(enc1,upconv1), upconv1), dim=1) 
        dec1 = self.dec1(skip_con1_add)

        #final output from the feature map with a 1 conv layer 
        return self.final_conv(dec1)