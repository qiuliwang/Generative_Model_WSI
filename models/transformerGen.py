
class ResnetVitGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        print('ResnetVitGenerator: ')
        assert(n_blocks >= 0)
        super(ResnetVitGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        for i in range(3):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # insert a Vit Block or other transformer-based block.
        '''
        self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.
        '''
        print('########## ngf * mult: ', ngf * mult)
        # model += nn.ModuleList(SwinTransformerBlock(3, (ngf * mult, ngf * mult), 3))
        hidden_dimension = 4
        num_heads = 3
        head_dim = 32
        window_size = 7
        relative_pos_embedding = True
        # for _ in range(2):
        #     model += nn.ModuleList([
        #         swim_transformer.SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
        #                   shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
        #         swim_transformer.SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
        #                   shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
        #     ])
        hidden_dim=96
        layers=(2, 2, 6, 2)
        heads=(3, 6, 12, 24)
        channels=3
        num_classes=3
        head_dim=32
        window_size=7
        downscaling_factors=(4, 2, 2, 2)
        relative_pos_embedding=True

        print('layers[0]: ', layers[0])
        print('layers type(): ', type(layers))

        model += swim_transformer.StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        

        for i in range(3):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
    def forward(self, input):
        """Standard forward"""
        print(input.size())
        out = self.model(input)
        print(out.size())
        return out