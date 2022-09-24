encode_model = torch.nn.Sequential(
    net.encoder,
    net.flatten,
    net.bottleneck[0],
    net.bottleneck[1],
    net.bottleneck[2]
)
