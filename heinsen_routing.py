import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakySoftmax(nn.Module):
    def __init__(self, dim):
        super(LeakySoftmax, self).__init__()
        self.dim = dim
        
    def forward(self, inp):
#             leak = torch.zeros_like(b_ij).sum(dim=2, keepdim=True)
#             leaky_logits = torch.cat((leak, b_ij),2)
#             leaky_routing = F.softmax(leaky_logits, dim=2)
#             c_ij = leaky_routing[:,:,1:,:].unsqueeze(4)
        
        
        maximum = torch.max(inp, self.dim, keepdim=True)
        power = torch.exp(inp - maximum)
        return power/torch.sum(power, dim=self.dim, keepdim=True)


class Routing(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "An
    Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.

    Args:
        d_cov: int, dimension 1 of input and output capsules.
        d_inp: int, dimension 2 of input capsules.
        d_out: int, dimension 2 of output capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_out: (optional) int, number of output capsules. If not provided, it
            can be passed to the forward method; otherwise it will be equal
            to the number of input capsules, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool; if True, beta_use and beta_ign are the
            same parameter, otherwise they are distinct. Default: False.
        p_model: (optional) str, specifies how to compute probability of input
            votes at each output capsule. Choices are 'gaussian' for Gaussian
            mixtures and 'skm' for soft k-means. Default: 'gaussian'.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores.
        mu_inp: [..., n_inp, d_cov, d_inp] capsules of shape d_cov x d_inp.
        return_R: (optional) bool, if True, return routing probabilities R
            in addition to other outputs. Default: False.
        n_out: (optional) int, number of output capsules. Valid as an input
            only if not already specified as an argument at initialization.

    Output:
        a_out: [..., n_out] output scores.
        mu_out: [..., n_out, d_cov, d_out] capsules of shape d_cov x d_out.
        sig2_out: [..., n_out, d_cov, d_out] variances of shape d_cov x d_out.

    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(mu_out)  # 10 capsules of shape 4 x 4
    """
    def __init__(self, d_cov, d_inp, d_out, n_inp=-1, n_out=-1, n_iters=3, single_beta=False, p_model='gaussian', eps=1e-5):
        super().__init__()
        assert p_model in ['gaussian', 'skm'], 'Unrecognized value for p_model.'
        self.n_iters, self.p_model, self.eps = (n_iters, p_model, eps)
        self.n_inp_is_fixed, self.n_out_is_fixed = (n_inp > 0, n_out > 0)
        one_or_n_inp, one_or_n_out = (max(1, n_inp), max(1, n_out))
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, one_or_n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out, d_cov, d_out))
        if not self.n_out_is_fixed: self.B_brk = nn.Parameter(torch.zeros(1, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.beta_ign = self.beta_use if single_beta else nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.f, self.log_f = (nn.Sigmoid(), nn.LogSigmoid())
        self.softmax, self.log_softmax = (nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1))

    def forward(self, a_inp, mu_inp, return_R=False, **kwargs):
        n_inp = a_inp.shape[-1]
        W = self.W if self.n_inp_is_fixed else self.W.expand(n_inp, -1, -1, -1)
        B = self.B
        if self.n_out_is_fixed:
            if ('n_out' in kwargs): raise ValueError('n_out is fixed!')
            n_out = W.shape[1]
        else:
            n_out = kwargs['n_out'] if ('n_out' in kwargs) else n_inp
            W = W.expand(-1, n_out, -1, -1)
            B = B + self.B_brk * torch.linspace(-1, 1, n_out, device=B.device)[:, None, None]  # break symmetry
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + B
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        if self.n_iters > 0:
            for iter_num in range(self.n_iters):

                # E-step.
                if iter_num == 0:
                    R = (self.CONST_one / n_out).expand(V.shape[:-2])  # [...ij]
                else:
                    log_p_simplified = \
                        - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                        - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2) if (self.p_model == 'gaussian') \
                        else self.log_softmax(-V_less_mu_out_2.sum((-2, -1)))  # soft k-means otherwise
                    R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

                # D-step.
                D_use = f_a_inp * R
                D_ign = f_a_inp - D_use

                # M-step.
                a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
                over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
                mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
                V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
                sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps
            ret_a = a_out
        else:
            R = (self.CONST_one / n_out).expand(V.shape[:-2])  # [...ij]
            
            D_use = f_a_inp * R
            D_ign = f_a_inp - D_use

            # M-step.
            a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
            over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
            mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps
            
            
#             last_a = a_out
#             loss = F.log_softmax(a_out, dim=-1)
            values, _ = torch.max(self.softmax(a_out), dim=1)
            last_a = torch.mean(values)
            ret_a = a_out

            count = 0
            while True and count < 7:
                count += 1
                    
                log_p_simplified = \
                    - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2) if (self.p_model == 'gaussian') \
                    else self.log_softmax(-V_less_mu_out_2.sum((-2, -1)))  # soft k-means otherwise
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

                # D-step.
                D_use = f_a_inp * R
                D_ign = f_a_inp - D_use

                # M-step.
                a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
                over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
                mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
                V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
                sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps
                
                values, _ = torch.max(self.softmax(a_out), dim=1)
                               
                candidate_a = torch.mean(values)
                
#                 print("last_a:", last_a)
#                 print("candidate_a:", candidate_a)

                # Also try
                # if candidate_a - last_a < 0:
                #     break

                # Can also try a version where we iterate til the max score goes down.
                if candidate_a > last_a: 
                    ret_a = a_out
#                 cur_loss = -target * F.log_softmax(a_out, dim=-1)
#                 print(cur_loss)
#                 print(loss)
                # TODO 0.05 is an arbritray epsilon decided by: https://github.com/andyweizhao/NLP-Capsule/blob/master/layer.py
                if candidate_a - last_a < 0.05 and candidate_a > (1/n_out + 0.05):
#                     a_out = ret_a
                    break
                else:
                    last_a = candidate_a                 
#         return (a_out, mu_out, sig2_out, R) if return_R else (a_out, mu_out, sig2_out)
        return (ret_a, mu_out, sig2_out, R) if return_R else (ret_a, mu_out, sig2_out)




class RoutingRNN(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "An
    Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.

    Args:
        d_cov: int, dimension 1 of input and output capsules.
        d_inp: int, dimension 2 of input capsules.
        d_out: int, dimension 2 of output capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_out: (optional) int, number of output capsules. If not provided, it
            can be passed to the forward method; otherwise it will be equal
            to the number of input capsules, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool; if True, beta_use and beta_ign are the
            same parameter, otherwise they are distinct. Default: False.
        p_model: (optional) str, specifies how to compute probability of input
            votes at each output capsule. Choices are 'gaussian' for Gaussian
            mixtures and 'skm' for soft k-means. Default: 'gaussian'.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores.
        mu_inp: [..., n_inp, d_cov, d_inp] capsules of shape d_cov x d_inp.
        return_R: (optional) bool, if True, return routing probabilities R
            in addition to other outputs. Default: False.
        n_out: (optional) int, number of output capsules. Valid as an input
            only if not already specified as an argument at initialization.

    Output:
        a_out: [..., n_out] output scores.
        mu_out: [..., n_out, d_cov, d_out] capsules of shape d_cov x d_out.
        sig2_out: [..., n_out, d_cov, d_out] variances of shape d_cov x d_out.

    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(mu_out)  # 10 capsules of shape 4 x 4
    """
    def __init__(self, d_cov, d_inp, d_out, n_inp=-1, n_out=-1, n_iters=3, single_beta=False, p_model='gaussian', eps=1e-5):
        super().__init__()
        assert p_model in ['gaussian', 'skm'], 'Unrecognized value for p_model.'
        self.n_iters, self.p_model, self.eps = (n_iters, p_model, eps)
        self.n_inp_is_fixed, self.n_out_is_fixed = (n_inp > 0, n_out > 0)
        one_or_n_inp, one_or_n_out = (max(1, n_inp), max(1, n_out))
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, one_or_n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out, d_cov, d_out))
        if not self.n_out_is_fixed: self.B_brk = nn.Parameter(torch.zeros(1, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.beta_ign = self.beta_use if single_beta else nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.f, self.log_f = (nn.Sigmoid(), nn.LogSigmoid())
        self.softmax, self.log_softmax = (nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1))
        # If this works abstract out into parameter
        self.hidden_size = 64
        self.rnnCell = nn.LSTMCell(n_out, self.hidden_size)
        self.output  = nn.Linear(self.hidden_size, n_out)
        

    def forward(self, a_inp, mu_inp, return_R=False, **kwargs):
        n_inp = a_inp.shape[-1]
        batch_size = a_inp.shape[0]
        W = self.W if self.n_inp_is_fixed else self.W.expand(n_inp, -1, -1, -1)
        B = self.B
        if self.n_out_is_fixed:
            if ('n_out' in kwargs): raise ValueError('n_out is fixed!')
            n_out = W.shape[1]
        else:
            n_out = kwargs['n_out'] if ('n_out' in kwargs) else n_inp
            W = W.expand(-1, n_out, -1, -1)
            B = B + self.B_brk * torch.linspace(-1, 1, n_out, device=B.device)[:, None, None]  # break symmetry
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + B
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]

        hidden = self.init_hidden(batch_size).cuda(device=B.device)
        cell   = self.init_hidden(batch_size).cuda(device=B.device)
        
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = (self.CONST_one / n_out).expand(V.shape[:-2])  # [...ij]
            else:
                log_p_simplified = \
                    - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2) if (self.p_model == 'gaussian') \
                    else self.log_softmax(-V_less_mu_out_2.sum((-2, -1)))  # soft k-means otherwise
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

            # D-step.
            D_use = f_a_inp * R
            D_ign = f_a_inp - D_use

            # M-step.
            a_temp = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)
#             a_feed = torch.cat((a_temp, a_inp),dim=1)
            hidden, cell = self.rnnCell(a_temp, (hidden, cell))
            
            a_out = self.output(hidden) 
            
#             a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
            over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
            mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps
            
        return (a_out, mu_out, sig2_out, R) if return_R else (a_out, mu_out, sig2_out)
    
    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return torch.zeros(bs, self.hidden_size)
    
    
class RoutingRNNCombo(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "An
    Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.

    Args:
        d_cov: int, dimension 1 of input and output capsules.
        d_inp: int, dimension 2 of input capsules.
        d_out: int, dimension 2 of output capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_out: (optional) int, number of output capsules. If not provided, it
            can be passed to the forward method; otherwise it will be equal
            to the number of input capsules, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool; if True, beta_use and beta_ign are the
            same parameter, otherwise they are distinct. Default: False.
        p_model: (optional) str, specifies how to compute probability of input
            votes at each output capsule. Choices are 'gaussian' for Gaussian
            mixtures and 'skm' for soft k-means. Default: 'gaussian'.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores.
        mu_inp: [..., n_inp, d_cov, d_inp] capsules of shape d_cov x d_inp.
        return_R: (optional) bool, if True, return routing probabilities R
            in addition to other outputs. Default: False.
        n_out: (optional) int, number of output capsules. Valid as an input
            only if not already specified as an argument at initialization.

    Output:
        a_out: [..., n_out] output scores.
        mu_out: [..., n_out, d_cov, d_out] capsules of shape d_cov x d_out.
        sig2_out: [..., n_out, d_cov, d_out] variances of shape d_cov x d_out.

    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(mu_out)  # 10 capsules of shape 4 x 4
    """
    def __init__(self, d_cov, d_inp, d_out, n_inp=-1, n_out=-1, n_iters=3, single_beta=False, p_model='gaussian', eps=1e-5):
        super().__init__()
        assert p_model in ['gaussian', 'skm'], 'Unrecognized value for p_model.'
        self.n_iters, self.p_model, self.eps = (n_iters, p_model, eps)
        self.n_inp_is_fixed, self.n_out_is_fixed = (n_inp > 0, n_out > 0)
        one_or_n_inp, one_or_n_out = (max(1, n_inp), max(1, n_out))
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, one_or_n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out, d_cov, d_out))
        if not self.n_out_is_fixed: self.B_brk = nn.Parameter(torch.zeros(1, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.beta_ign = self.beta_use if single_beta else nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.f, self.log_f = (nn.Sigmoid(), nn.LogSigmoid())
        self.softmax, self.log_softmax = (nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1))
        # If this works abstract out into parameter
        self.hidden_size = 128
        self.rnnCell = nn.LSTMCell(n_out + 4*(n_out), self.hidden_size)
        self.output  = nn.Linear(self.hidden_size, n_out)
        

    def forward(self, a_inp, mu_inp, return_R=False, **kwargs):
        n_inp = a_inp.shape[-1]
        batch_size = a_inp.shape[0]
        W = self.W if self.n_inp_is_fixed else self.W.expand(n_inp, -1, -1, -1)
        B = self.B
        if self.n_out_is_fixed:
            if ('n_out' in kwargs): raise ValueError('n_out is fixed!')
            n_out = W.shape[1]
        else:
            n_out = kwargs['n_out'] if ('n_out' in kwargs) else n_inp
            W = W.expand(-1, n_out, -1, -1)
            B = B + self.B_brk * torch.linspace(-1, 1, n_out, device=B.device)[:, None, None]  # break symmetry
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + B
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]

        hidden = self.init_hidden(batch_size).cuda(device=B.device)
        cell   = self.init_hidden(batch_size).cuda(device=B.device)
        
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = (self.CONST_one / n_out).expand(V.shape[:-2])  # [...ij]
            else:
                log_p_simplified = \
                    - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2) if (self.p_model == 'gaussian') \
                    else self.log_softmax(-V_less_mu_out_2.sum((-2, -1)))  # soft k-means otherwise
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

            # D-step.
            D_use = f_a_inp * R
            D_ign = f_a_inp - D_use

            # M-step.
            a_temp = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)
#             a_feed = torch.cat((a_temp, a_inp),dim=1)

            
#             a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
            over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
            mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps

            a_feed = torch.cat((a_temp, mu_out.reshape(batch_size, -1), sig2_out.reshape(batch_size, -1)),dim=1)
            
            hidden, cell = self.rnnCell(a_feed, (hidden, cell))
            
            a_out = self.output(hidden) 
             
        return (a_out, mu_out, sig2_out, R) if return_R else (a_out, mu_out, sig2_out)
    
    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return torch.zeros(bs, self.hidden_size)
    
    
class RoutingRNNLearnedRouting(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "An
    Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.

    Args:
        d_cov: int, dimension 1 of input and output capsules.
        d_inp: int, dimension 2 of input capsules.
        d_out: int, dimension 2 of output capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_out: (optional) int, number of output capsules. If not provided, it
            can be passed to the forward method; otherwise it will be equal
            to the number of input capsules, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool; if True, beta_use and beta_ign are the
            same parameter, otherwise they are distinct. Default: False.
        p_model: (optional) str, specifies how to compute probability of input
            votes at each output capsule. Choices are 'gaussian' for Gaussian
            mixtures and 'skm' for soft k-means. Default: 'gaussian'.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores.
        mu_inp: [..., n_inp, d_cov, d_inp] capsules of shape d_cov x d_inp.
        return_R: (optional) bool, if True, return routing probabilities R
            in addition to other outputs. Default: False.
        n_out: (optional) int, number of output capsules. Valid as an input
            only if not already specified as an argument at initialization.

    Output:
        a_out: [..., n_out] output scores.
        mu_out: [..., n_out, d_cov, d_out] capsules of shape d_cov x d_out.
        sig2_out: [..., n_out, d_cov, d_out] variances of shape d_cov x d_out.

    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(mu_out)  # 10 capsules of shape 4 x 4
    """
    def __init__(self, d_cov, d_inp, d_out, n_inp=-1, n_out=-1, n_iters=3, single_beta=False, p_model='gaussian', eps=1e-5):
        super().__init__()
        assert p_model in ['gaussian', 'skm'], 'Unrecognized value for p_model.'
        self.n_iters, self.p_model, self.eps = (n_iters, p_model, eps)
        self.n_inp_is_fixed, self.n_out_is_fixed = (n_inp > 0, n_out > 0)
        one_or_n_inp, one_or_n_out = (max(1, n_inp), max(1, n_out))
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, one_or_n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out, d_cov, d_out))
        if not self.n_out_is_fixed: self.B_brk = nn.Parameter(torch.zeros(1, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.beta_ign = self.beta_use if single_beta else nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.f, self.log_f = (nn.Sigmoid(), nn.LogSigmoid())
        self.softmax, self.log_softmax = (nn.Softmax(dim=-1), nn.LogSoftmax(dim=-1))
        # If this works abstract out into parameter
        self.a_scaler = nn.Linear(n_out, n_out)
        self.mu_scaler = nn.Linear(n_out*4, n_out*2)
        
    def forward(self, a_inp, mu_inp, return_R=False, **kwargs):
        n_inp = a_inp.shape[-1]
        batch_size = a_inp.shape[0]
        W = self.W if self.n_inp_is_fixed else self.W.expand(n_inp, -1, -1, -1)
        B = self.B
        if self.n_out_is_fixed:
            if ('n_out' in kwargs): raise ValueError('n_out is fixed!')
            n_out = W.shape[1]
        else:
            n_out = kwargs['n_out'] if ('n_out' in kwargs) else n_inp
            W = W.expand(-1, n_out, -1, -1)
            B = B + self.B_brk * torch.linspace(-1, 1, n_out, device=B.device)[:, None, None]  # break symmetry
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + B
        f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
        
        R = (self.CONST_one / n_out).expand(V.shape[:-2])
        
        D_use = f_a_inp * R
        D_ign = f_a_inp - D_use

        # M-step.
        a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
        
        over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
        mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
        V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
        sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps
        
        
        mu_shape = mu_out.shape
        
        ins = torch.cat((mu_out.reshape(batch_size,-1), sig2_out.reshape(batch_size,-1)), dim=1)
        
        a_out = self.f(self.a_scaler(a_out))
        mu_out = mu_out * self.f(self.mu_scaler(ins)).reshape(mu_shape)
        
             
        return (a_out, mu_out, sig2_out, R) if return_R else (a_out, mu_out, sig2_out)