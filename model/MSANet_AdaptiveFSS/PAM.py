import torch
import torch.nn as nn
import torch.nn.functional as F

class Prototype_Adaptive_Module(nn.Module):
      """ 
      Protoype adaptive module for query-support feature enhancement.
      produce enhanced features for Few-shot segmentation task.
      Args:
            dim (int): Number of input channels.
            input_resolution (tuple): input_resolution for high and weight
            hidden_ratio(float): the compressed ratio of feature dim
            class_num(int): Number of adaptive class
            momentum(float): the momentum updating ratio for prototype
      """
      def __init__(self, 
                   dim, 
                   input_resolution, 
                   hidden_ratio=16., 
                   drop=0.,
                   class_num=1.,
                   momentum=0.9):
            super().__init__()
            self.dim = dim
            self.dim_low = dim // hidden_ratio
            self.input_resolution = input_resolution
            self.act = nn.ReLU()
            self.linears_down = nn.Linear(dim, self.dim_low)
            self.linears_up = nn.Linear(self.dim_low, dim)

            self.register_buffer("prototype", torch.randn(dim, class_num).requires_grad_(False))
            self.prototype = nn.functional.normalize(self.prototype, dim=0)

            self.drop = drop
            self.proj_drop = nn.Dropout(drop)
            self.momentum = momentum
            self.act_enhance = nn.ReLU6()

      def forward(self, x, s_f, s_y, class_idx=None):
            """ 
            Args:
            x: torch.Tensor
                [B , h * w, dim], query 
            s_f: torch.Tensor
                [B * S , h * w, dim], support 
            s_y: torch.Tensor
                [B, S, H, W],    support mask one-shot
            class_id: list
                len(class_id) = B
            Outputs:
            registered_feas: torch.Tensor
                [B * (S + 1), h * w, dim], injected query features
            """
            B, N, D = x.shape
            s_f = s_f.reshape(B, -1, N, D)
            S = s_f.size(1)

            s_y = F.interpolate(s_y, (self.input_resolution[0], self.input_resolution[1]), mode='nearest')
            sup_mask_fg = (s_y == 1).float().reshape(B, S, -1).reshape(B, -1).unsqueeze(-1) #[B, S * N, 1]
            semantic_prototype, sign_fore_per_batch = self.extract_semantic_prototype(s_f, sup_mask_fg)

            if class_idx is not None:
                with torch.no_grad():
                    self.train()
                    new_semantic_prototype = self.updata_prototype_bank(semantic_prototype, class_idx, sign_fore_per_batch)

            else:
                self.eval()
                new_semantic_prototype = self.select_prototype_bank(semantic_prototype, self.prototype)
            enhanced_feat_q = self.enhanced_feature(x.unsqueeze(1), new_semantic_prototype, sign_fore_per_batch)
            enhanced_feat_sup = self.enhanced_feature(s_f, new_semantic_prototype, sign_fore_per_batch)

            registered_feas = torch.cat([enhanced_feat_sup.reshape(-1, N, D), enhanced_feat_q.squeeze(1)], dim=0)
            registered_feas = self.proj_drop(self.linears_up(self.act(self.linears_down(registered_feas))))

            return registered_feas

      def extract_semantic_prototype(self, s_f, s_y):
          """
          extract temporary class prototype according to support features and masks
          input:
            s_f: torch.Tensor
                [B, S, N, D], support features
            s_y: torch.Tensor
                [B, S * N, 1], support masks
          output:
            semantic_prototype: torch.Tensor
                [B, D], temporary prototypes
            sign_fore_per_batch: torch.Tensor
                [B], the signal of whether including foreground region in this image
          """
          B, S, N, D = s_f.shape
          num_fore_per_batch = torch.count_nonzero(s_y.reshape(B, -1), dim=1)
          s_y = s_y.repeat(1, 1, D)
          semantic_prototype = s_y * s_f.reshape(B, -1, D)
          semantic_prototype = semantic_prototype.mean(1) * (N * S) / (num_fore_per_batch.unsqueeze(1)+1e-4)
          one = torch.ones_like(num_fore_per_batch).cuda()
          sign_fore_per_batch =  torch.where(num_fore_per_batch > 0.5, one, num_fore_per_batch)
          return semantic_prototype.reshape(B,D), sign_fore_per_batch
      
      def updata_prototype_bank(self, semantic_prototype, class_idx, sign_fore_per_batch):
          """
          updata prototype in class prototype bank during traning
          input:
            semantic_prototype: torch.Tensor
                [B, D]
            class_id: list
                len(class_id) = B
            sign_fore_per_batch: torch.Tensor
                [B], the signal of whether including foreground region in this image
          output:
            new_semantic_prototype: torch.Tensor
                [B, D], the updated prototypes for feature enhancement
          """   
          B, D = semantic_prototype.shape
          self.prototype = nn.functional.normalize(self.prototype, dim=0)
          semantic_prototype = nn.functional.normalize(semantic_prototype, dim=1)
          new_semantic_prototype_list = []
          for i in range(B):
               semantic_prototype_per = semantic_prototype[i,: ]
               class_idx_per = class_idx[i]
               if sign_fore_per_batch[i] == 1:
                    new_semantic_prototype_per = self.prototype[:, class_idx_per] * self.momentum + (1 - self.momentum) * semantic_prototype_per
                    self.prototype[:, class_idx_per] = new_semantic_prototype_per
               else: 
                    new_semantic_prototype_per = self.prototype[:, class_idx_per]    
               new_semantic_prototype_list.append(new_semantic_prototype_per)
          new_semantic_prototype = torch.stack(new_semantic_prototype_list, dim=0)
          return new_semantic_prototype.reshape(B,D)

      def select_prototype_bank(self, semantic_prototype, prototype_bank):
          """
          select prototypes in class prototype bank during testing
          input:
            semantic_prototype: torch.Tensor
                shape = [B, D]
            prototype_bank: torch.Tensor
                shape = [D, class_num]
          output:
            new_semantic_prototype: torch.Tensor
                [B, D], the prototypes for feature enhancement
          """ 
          B, D = semantic_prototype.shape
          prototype_bank = nn.functional.normalize(prototype_bank, dim=0)
          semantic_prototype = nn.functional.normalize(semantic_prototype, dim=1)
          similar_matrix = semantic_prototype @ prototype_bank  # [B, class_num]
          idx = similar_matrix.argmax(1)

          new_semantic_prototype_list = []
          for i in range(B):
              new_semantic_prototype_per = prototype_bank[:, idx[i]]
              new_semantic_prototype_list.append(new_semantic_prototype_per)
          new_semantic_prototype = torch.stack(new_semantic_prototype_list, dim=0)
          return new_semantic_prototype.reshape(B,D)

      def enhanced_feature(self, feature, new_semantic_prototype, sign_fore_per_batch):
          """ 
            Input:
                feature: torch.Tensor
                    [B, S, N, D]
                new_semantic_prototype: torch.Tensor
                    [B, D]
            Outputs:
                enhanced_feature: torch.Tensor
                    [B, S, N, D]
            """
          B, D = new_semantic_prototype.shape
          feature_sim = nn.functional.normalize(feature, p=2, dim=-1)
          new_semantic_prototype = nn.functional.normalize(new_semantic_prototype, p=2, dim=1)
          similarity_matrix_list = []
          for i in range(B):
              feature_sim_per = feature_sim[i,:, :, :]
              new_semantic_prototype_er = new_semantic_prototype[i, :]
              similarity_matrix_per = feature_sim_per @ new_semantic_prototype_er
              similarity_matrix_list.append(similarity_matrix_per)
          similarity_matrix = torch.stack(similarity_matrix_list, dim=0)

          similarity_matrix = (similarity_matrix * self.dim ** 0.5) *  sign_fore_per_batch.unsqueeze(-1).unsqueeze(-1)

          enhanced_feature = self.act_enhance(similarity_matrix).unsqueeze(-1).repeat(1, 1, 1, D) * feature + feature


          return enhanced_feature