# -*- coding: utf-8 -*-
"""
GNN: NodeEncoder объединяет тег+class (структура) отдельным Dense, затем конкатенирует с текстом; SAGEConv.

Лучшая модель диплома — BiDirSAGEStabV3 (двунаправленный GraphSAGE + stability-признаки
с регуляризациями DropEdge и Feature Dropout). Остальные архитектуры экспериментов
(однонаправленная, без регуляризаций, абляции) в этот репозиторий не вошли.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge


class NodeEncoder(nn.Module):
    """
    Кодировщик узла DOM:
      1. e_tag  = Embedding(tag_index)                     [embed_dim]
      2. e_class = proj_class(FastText avg class words)    [embed_dim]
         (нулевой вектор если class отсутствует)
      3. h_tag_class = ReLU(tag_class_merge([e_tag, e_class]))  [embed_dim]
      4. h_text = proj_text(FastText visible text)         [embed_dim]
      5. h_textual = merge([h_tag_class, h_text])          [hidden_dim]
      6. h_num  = proj_num(числовые признаки)              [hidden_dim]
      7. out = h_textual + h_num
    """

    def __init__(
        self,
        ft_dim: int,
        embed_dim: int,
        num_tag_embeddings: int,
        num_numeric: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.tag_embed = nn.Embedding(num_tag_embeddings, embed_dim)
        self.proj_class = nn.Linear(ft_dim, embed_dim)
        # тег + class → Dense → один структурный вектор
        self.tag_class_merge = nn.Linear(2 * embed_dim, embed_dim)
        self.proj_text = nn.Linear(ft_dim, embed_dim)
        # структурный вектор + текстовый → hidden
        self.merge = nn.Linear(2 * embed_dim, hidden_dim)
        self.proj_num = nn.Linear(num_numeric, hidden_dim)

    def forward(
        self,
        x_tag: torch.Tensor,
        x_text: torch.Tensor,
        x_class: torch.Tensor,
        x_num: torch.Tensor,
    ) -> torch.Tensor:
        e_tag = self.tag_embed(x_tag.long())
        e_class = self.proj_class(x_class)
        h_tag_class = F.relu(self.tag_class_merge(torch.cat([e_tag, e_class], dim=-1)))
        h_text = F.relu(self.proj_text(x_text))
        h_textual = F.relu(self.merge(torch.cat([h_tag_class, h_text], dim=-1)))
        h_num = F.relu(self.proj_num(x_num))
        return h_textual + h_num


# ---------------------------------------------------------------------------
# Лучшая модель: BiDir + stability + DropEdge + Feature Dropout (stab-блок)
# ---------------------------------------------------------------------------

class BiDirSAGEStabV3(nn.Module):
    """
    Двунаправленный GraphSAGE поверх DOM-дерева. Каждый слой делает два прохода:
      - восходящий  (child→parent, edge_index)
      - нисходящий  (parent→child, flip(edge_index))
    результаты конкатенируются и проецируются обратно в hidden_dim.

    Две регуляризации (активны только в train-режиме, на инференсе не выполняются):
      1. DropEdge(p=drop_edge_p) — на каждом forward в train-режиме случайно отбрасывается
         доля рёбер edge_index. Одна и та же mask применяется к up- и down-направлениям
         (симметричный DropEdge), чтобы оба прохода видели согласованную структуру.
      2. Feature Dropout(p=feat_drop_p) на stab-блок — последние stab_dim столбцов x_num
         рандомно зануляются перед энкодером в train-режиме. Эмулирует ситуацию, когда у
         узла нет stab-сигнала; учит модель работать и при наличии, и при отсутствии counters.
    """

    def __init__(
        self,
        ft_dim: int,
        embed_dim: int,
        num_tag_embeddings: int,
        num_numeric: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        stab_dim: int = 5,
        drop_edge_p: float = 0.1,
        feat_drop_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.stab_dim = stab_dim
        self.drop_edge_p = drop_edge_p
        self.feat_drop_p = feat_drop_p

        self.encoder = NodeEncoder(
            ft_dim, embed_dim, num_tag_embeddings, num_numeric, hidden_dim
        )
        self.dropout = nn.Dropout(dropout)

        self.up_convs   = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.down_convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.projs = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x_tag: torch.Tensor,
        x_text: torch.Tensor,
        x_class: torch.Tensor,
        x_num: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Feature Dropout на stab-блок (последние stab_dim столбцов).
        if self.training and self.feat_drop_p > 0.0 and self.stab_dim > 0 and x_num.size(1) >= self.stab_dim:
            x_num_base = x_num[:, :-self.stab_dim]
            x_num_stab = F.dropout(x_num[:, -self.stab_dim:], p=self.feat_drop_p, training=True)
            x_num = torch.cat([x_num_base, x_num_stab], dim=-1)

        # DropEdge: одна mask на все слои, симметрично для up и down.
        if self.training and self.drop_edge_p > 0.0 and edge_index.size(1) > 0:
            edge_index_up, _ = dropout_edge(edge_index, p=self.drop_edge_p, training=True)
        else:
            edge_index_up = edge_index
        edge_index_down = edge_index_up[[1, 0], :] if edge_index_up.size(1) > 0 else edge_index_up

        h = F.relu(self.encoder(x_tag, x_text, x_class, x_num))

        for up_conv, down_conv, proj, bn in zip(self.up_convs, self.down_convs, self.projs, self.bns):
            h_res = h
            h_up   = up_conv(h, edge_index_up)
            h_down = down_conv(h, edge_index_down)
            h_merged = proj(torch.cat([h_up, h_down], dim=-1))
            h = self.dropout(F.relu(bn(h_merged))) + h_res

        return self.head(h)


def build_stability_bidir_v3_model(
    ft_dim: int,
    num_numeric: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_tag_embeddings: int,
    stability_dim: int = 5,
    drop_edge_p: float = 0.1,
    feat_drop_p: float = 0.2,
    num_classes: int = 2,
) -> BiDirSAGEStabV3:
    """
    BiDir + stability с регуляризациями DropEdge и Feature Dropout (stab).
    num_numeric — базовое число числовых признаков (из sage_preprocessors.pt).
    stability_dim — число признаков стабильности (вход в feat-dropout-маску).
    """
    return BiDirSAGEStabV3(
        ft_dim=ft_dim,
        embed_dim=embed_dim,
        num_tag_embeddings=num_tag_embeddings,
        num_numeric=num_numeric + stability_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        stab_dim=stability_dim,
        drop_edge_p=drop_edge_p,
        feat_drop_p=feat_drop_p,
    )
