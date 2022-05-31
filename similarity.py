from models.resnet_base_network import ResNet18
import torch
import random

class SimFilter:
    def __init__(self, args, encoder=None):
        self.args = args
        self.gpu = 0
        if args.sim_pretrained:
            self.encoder = self.load_pretrained_encoder()
        else:
            self.encoder = encoder 

    # Load ResNet18 for similarity check
    def load_pretrained_encoder(self):
        model = ResNet18(name='resnet18', hidden_dim=512, proj_size=128)
        state_dict = {}
        for k, v in torch.load('./teacher.pth')['online_network_state_dict'].items():
            new_key = k.replace("projetion", "projection")[7:]
            state_dict[new_key] = v
        model.load_state_dict(state_dict)
        model = model.to(self.args.gpu)
        model.eval()
        return model

    # Calculate the Euclidian Distance between the representation vector from pre-trained Encoder(ResNet18) and Projector(MLP)
    @torch.no_grad()
    def calc_l2_similarity(self, img_pair_list):
        v1_list = self.encoder(img_pair_list[0])
        v2_list = self.encoder(img_pair_list[1])
        ret = [torch.dist(v1, v2) for v1, v2 in zip(v1_list, v2_list)]
        return ret

    # Calculate the Cosine Distance between the representation vector from pre-trained Encoder(ResNet18) and Projector(MLP)
    @torch.no_grad()
    def calc_cos_similarity(self, img_pair_list):
        v1_list = self.encoder(img_pair_list[0])
        v2_list = self.encoder(img_pair_list[1])
        return torch.nn.functional.cosine_similarity(v1_list, v2_list)

    # Sort the image batch according to similarity (batch structure: list[[aug1, aug2], [label]])
    def sort_by_similarity(self, batch):
        # similarity = self.calc_l2_similarity(batch[0])
        similarity = self.calc_cos_similarity(batch[0])
        sorted_batch = sorted(zip(batch[0][0], batch[0][1], batch[1], similarity), key=lambda x: x[3], reverse=False)
        return [[[x[0] for x in sorted_batch], [x[1] for x in sorted_batch]], [x[2] for x in sorted_batch]]

    def sort_by_similarity_each(self, view1, view2, reverse=False):
        similarity = self.calc_cos_similarity([view1, view2])
        
        print(similarity.shape)
        print(similarity)
        sorted_batch = sorted(zip(view1, view2, similarity), key= lambda x:x[2], reverse=(not reverse))
        return [x[0] for x in sorted_batch], [x[1] for x in sorted_batch]

    # Sort the image batch and selecect specific ratio according to similarity (batch structure: list[[aug1, aug2], [label]])
    def filter_by_similarity_ratio(self, batch_view_1, batch_view_2, epoch, rep_1 = None, rep_2 = None):
        
        # similarity = self.calc_l2_similarity(batch[0])
        bool_rep = rep_1 is not None and rep_2 is not None
        if bool_rep:
            with torch.no_grad():
                similarity = torch.nn.functional.cosine_similarity(rep_1, rep_2)
        else:
            similarity = self.calc_cos_similarity([batch_view_1, batch_view_2])
        
        sorted_batch = sorted(zip(batch_view_1, batch_view_2, similarity), key=lambda x: x[2], reverse=True)

        # Remove outliers (3(sigma))
        processed_batch = sorted_batch[self.args.sigma3 : -self.args.sigma3]

        # print(f"len(processed_batch) : {len(processed_batch)}")
        reduce = len(processed_batch) - self.args.orig_batch_size
        begin = int((len(processed_batch) - self.args.orig_batch_size) * epoch / self.args.max_epochs)
        end = reduce - begin

        # print(f"reduce : {reduce}")
        # print(f"begin : {begin}")
        # print(f"end : {end}")
        processed_batch = processed_batch[begin : -end]

        v1_list = []
        v2_list = []
        for v1, v2, _ in processed_batch:
            v1_list.append(v1)
            v2_list.append(v2)
        
        if bool_rep:
            sorted_rep = sorted(zip(rep_1, rep_2, similarity), key=lambda x: x[2], reverse=True)
            if self.args.filter_type == 'window':
                processed_rep = sorted_rep[self.args.sigma3 + begin: -(self.args.sigma3 + end)]
            elif self.args.filter_type == 'sample':
                processed_rep = random.sample(sorted_rep[self.args.sigma3 : -(self.args.sigma3 + end)], self.args.orig_batch_size)

            rep_v1_list = []
            rep_v2_list = []
            for v1, v2, _ in processed_rep:
                rep_v1_list.append(v1)
                rep_v2_list.append(v2)
            del sorted_rep, processed_rep
            return torch.stack(v1_list), torch.stack(v2_list), torch.stack(rep_v1_list), torch.stack(rep_v2_list)

        return torch.stack(v1_list), torch.stack(v2_list)
        
        #return ([[x[0] for x in processed_batch], [x[1] for x in processed_batch]])

    def check_if_two_square(self, number):
        while number:
            if number % 2 == 1:
                return False
            number = number / 2
        return True


if __name__ == "__main__":
    print("hi")