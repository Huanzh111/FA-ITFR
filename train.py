import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

  ##########################################################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)
    ##########################################################################################
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    
    
    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        loss_list = []
        image_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                # torch.Size([8, 768]) torch.Size([8, 1370, 768])
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).to(device)
                #########################################################################
            prompts, tokenized_prompts, prompts_class, tokenized_prompts_class, compound_prompts_text = prompt_learner(
                cls_id=None)

            text_features_class = model.encode_text_learn(prompts_class, tokenized_prompts_class,
                                                          compound_prompts_text).float()
            text_features_class_yuan = torch.stack(torch.chunk(text_features_class, dim=0, chunks=2), dim=1)
            text_features_class_yuan = text_features_class_yuan / text_features_class_yuan.norm(dim=-1, keepdim=True)

            # prompts: [2, 77, 768]; tokenized_prompts: [2, 77] # compound_prompts_text: 9 [4, 768]
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features_yuan = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features_yuan = text_features_yuan / text_features_yuan.norm(dim=-1, keepdim=True)

            text_probs_class = image_features.unsqueeze(1) @ text_features_class_yuan.permute(0, 2, 1)
            text_probs_class = text_probs_class[:, 0, ...] / 0.07
            image_loss_class = F.cross_entropy(text_probs_class.squeeze(1), label.long().cuda())

            text_probs = image_features.unsqueeze(1) @ text_features_yuan.permute(0, 2, 1)
            # text_probs: torch.Size([8, 1, 2])  image_features: torch.Size([8, 1, 768])
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(1), label.long().cuda())

            #########################################################################
            similarity_map_list = []
            class_map_list = []
            # similarity_map_list.append(similarity_map)
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features_yuan[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

                    similarity_class, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features_class_yuan[0])
                    similarity_map_class = AnomalyCLIP_lib.get_similarity_map(similarity_class[:, 1:, :],
                                                                            args.image_size).permute(0, 3, 1, 2)
                    class_map_list.append(similarity_map_class)

            loss = 0
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss_class = 0
            for i in range(len(class_map_list)):
                loss_class = loss_class + loss_focal(class_map_list[i], gt)
                loss_class = loss_class + loss_dice(class_map_list[i][:, 1, :, :], gt)
                loss_class = loss_class + loss_dice(class_map_list[i][:, 0, :, :], 1 - gt)

            image_loss_total = image_loss + image_loss_class * 0.000001
            # * 0.00001
            pixel_loss = loss + loss_class

            optimizer.zero_grad()
            (pixel_loss+image_loss_total).backward()
            optimizer.step()
            loss_list.append(pixel_loss.item())
            image_loss_list.append(image_loss_total.item())
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./visa/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./mvtec_train', help='path to save results')
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=16, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
