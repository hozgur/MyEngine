encoder = torch.nn.Sequential(
    net.encoder,
    net.flatten,
    net.bottleneck[0],
    net.bottleneck[1],
    net.bottleneck[2],
    net.bottleneck[3]
)
decoder = torch.nn.Sequential(
    net.bottleneck[4],
    net.bottleneck[5],
    net.bottleneck[6],
    net.bottleneck[7],
    net.unflatten,
    net.decoder
)
