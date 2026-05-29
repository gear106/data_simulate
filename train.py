import pytorch_lightning as pl
import yaml
import torch
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Model
from dataloader import DataModule


def main(args):
    pl.seed_everything(3407)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = TensorBoardLogger(save_dir=config['log_dir'], name='tensorboard')
    ckpt_dir = Path(config['log_dir']) / f'ckpts/version_{logger.version}' #change your folder, where to save files
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config['ckpt_dir'] = ckpt_dir
    model = Model(config=config)

    # 新任务微调：加载预训练权重，不恢复优化器状态
    if config.get('ckpt_path') is not None:
        ckpt = torch.load(config['ckpt_path'], map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict']

        # 先删除所有形状不匹配的键，避免 load_state_dict 报错
        model_state = model.state_dict()
        for k in list(state_dict.keys()):
            if k in model_state and state_dict[k].shape != model_state[k].shape:
                print(f"Skip loading {k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(model_state[k].shape)}")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)

        # 单独处理 task_embedding：预训练是3任务，微调是1任务（se），只取第0行
        if 'dnn.task_embedding.weight' in ckpt['state_dict']:
            pretrained_task_emb = ckpt['state_dict']['dnn.task_embedding.weight']  # [3, 512]
            if model.dnn.task_embedding.weight.shape[0] == 1 and pretrained_task_emb.shape[0] == 3:
                model.dnn.task_embedding.weight.data.copy_(pretrained_task_emb[0:1])
                print("Adapted task_embedding from 3 tasks to 1 task (se)")

        print(f"Loaded pretrained weights from {config['ckpt_path']}")

    data_module = DataModule(**config['dataset_config'])
    checkpoint_callback_last = ModelCheckpoint(dirpath=ckpt_dir, save_on_train_epoch_end=True, filename='{epoch}-last')
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        max_epochs=config['max_epochs'],
        val_check_interval=config['val_check_interval'],
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=[checkpoint_callback_last],
        logger=logger,
        strategy="auto" if len(config['devices']) == 1 else 'ddp_find_unused_parameters_true',
    )

    trainer.fit(model, data_module, ckpt_path=config['resume'])



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    args = parser.parse_args()
    main(args)