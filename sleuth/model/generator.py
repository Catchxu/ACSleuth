from .block import Extractor, MemoryBlock, StyleBlock


class GeneratorAD(Extractor):
    def __init__(self, 
                 in_dim, 
                 hidden_dim=[512, 256], 
                 num_blocks=2,
                 mem_dim=512, 
                 threshold=0.01, 
                 temperature=0.05):
        super().__init__(in_dim, hidden_dim, num_blocks)
        self.Memory = MemoryBlock(mem_dim, hidden_dim[-1], threshold, temperature)
        self.z_dim = hidden_dim[-1]

    def forward(self, x):
        z = self.Encoder(x)
        x = self.Decoder(self.Memory(z))
        return x, z
    
    def prepare(self, x):
        z = self.Encoder(x)
        x = self.Decoder(z)
        return x, z




class GeneratorDA(Extractor):
    def __init__(self, 
                 num_batches, 
                 in_dim, 
                 hidden_dim=[512, 256], 
                 num_blocks=2):
        super().__init__(in_dim, hidden_dim, num_blocks)
        self.Style = StyleBlock(num_batches, hidden_dim[-1])

    def forward(self, x):
        z = self.Encoder(x)
        x = self.Decoder(self.Style(z))
        return x