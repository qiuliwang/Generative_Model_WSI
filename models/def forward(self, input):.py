 def forward(self, input):
        """Standard forward"""
       #  print(input.size())
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print('layer3', x.size())

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # print('layer6', x.size())

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        # print('layer9', x.size())

        x = self.layer13(x)
        # print('layer13', x.size())


        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        # print('layer12', x.size())

        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        # print('layer16', x.size())


        
        # ConvTranspose2d
        x = self.layer13_1(x)
        x = self.layer13_2(x)
        x = self.layer13_3(x)
        # print('layer13_3', x.size())

        x = self.layer17(x)
        x = self.layer17_1(x)
        x = self.layer17_2(x)
        # print('layer17_2', x.size())

        x = self.layer18(x)
        x = self.layer18_1(x)
        x = self.layer18_2(x)
        # print('layer18_2', x.size())

        x = self.layer19(x)
        x = self.layer20(x)
        out = self.layer21(x)
        # print('layer21', out.size())

        return out