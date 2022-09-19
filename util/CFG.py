class CFG:
    def __init__(self):
        self.competition='FB3'
        self.print_freq=30
        self.model="microsoft/deberta-v3-base"
        self.gradient_checkpointing=True
        self.epochs=4
        self.encoder_lr=2e-5
        self.decoder_lr=2e-5
        self.min_lr=1e-6
        self.batch_size=8
        self.target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    
    def debug(self):
        self.epochs = 2
        self.trn_fold = [0]