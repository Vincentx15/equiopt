import os
import gzip
import random
import numpy as np
import torch

from seqdataloader_custom import PyfaidxCoordsToVals, SimpleLookup, \
    DownsampleNegativesCoordsBatchProducer, Coordinates, apply_mask
from torch.utils.data import Dataset, DataLoader


def get_revcomp(coordinate):
    return Coordinates(chrom=coordinate.chrom,
                       start=coordinate.start, end=coordinate.end,
                       isplusstrand=(coordinate.isplusstrand is False))


class AbstractCoordBatchTransformer(object):

    def __call__(self, coords):
        """
        Args:
            coords (:obj:`list` of :obj:`Coordinates` objects):

        Returns:
            another :obj:`list` of :obj:`Coordinates`
        """
        raise NotImplementedError()

    def chain(self, coord_batch_transformer):
        return lambda coords: coord_batch_transformer(self(coords))


class ReverseComplementAugmenter(AbstractCoordBatchTransformer):
    """
        Returns a list of Coordinates twice the length of the
            original list by appending the reverse complements
            of the original coordinates at the end
    """

    def __call__(self, coords):
        return coords + [get_revcomp(x) for x in coords]


# class RevcompTackedOnSimpleCoordsBatchProducer(DownsampleNegativesCoordsBatchProducer):
#     def _get_coordslist(self):
#         self.last_used_offset += 1
#         self.last_used_offset = self.last_used_offset % self.subsample_factor
#         print("Using an offset of ", self.last_used_offset, " before striding")
#         self.last_used_offset = self.last_used_offset % self.subsample_factor
#         subsampled_neg_coords = self.neg_bedfileobj.get_strided_subsample(
#             offset=self.last_used_offset,
#             stride=self.subsample_factor)
#         pos_coords = self.pos_bedfileobj.coords_list
#         self.subsampled_neg_coords = subsampled_neg_coords
#         self.pos_coords = pos_coords
#         curr_coords = pos_coords + subsampled_neg_coords
#         return [x for x in curr_coords] + [get_revcomp(x) for x in curr_coords]


class BatchDataset(Dataset):

    def __init__(self, TF, seq_len, is_aug,
                 seed=0, batch_size=100, ):

        inputs_coordstovals = PyfaidxCoordsToVals(
            genome_fasta_path="data/hg19.genome.fa",
            center_size_to_use=seq_len)

        targets_coordstovals = SimpleLookup(
            lookup_file=f"data/{TF}/{TF}_lookup.bed.gz",
            transformation=None,
            default_returnval=0.0)

        target_proportion_positives = 1 / 5

        pos_bed = f"data/{TF}/{TF}_foreground_train.bed.gz"
        neg_bed = f"data/{TF}/{TF}_background_train.bed.gz"

        coords_batch_producer = DownsampleNegativesCoordsBatchProducer(
            pos_bed_file=pos_bed,
            neg_bed_file=neg_bed,
            target_proportion_positives=target_proportion_positives,
            batch_size=batch_size,
            shuffle_before_epoch=True,
            seed=seed)
        coordsbatch_transformer = ReverseComplementAugmenter() if is_aug else None

        self.coordsbatch_producer = coords_batch_producer
        self.inputs_coordstovals = inputs_coordstovals
        self.targets_coordstovals = targets_coordstovals
        self.coordsbatch_transformer = coordsbatch_transformer

    def __getitem__(self, index):
        coords_batch = self.coordsbatch_producer[index]
        if self.coordsbatch_transformer is not None:
            coords_batch = self.coordsbatch_transformer(coords_batch)
        inputs = self.inputs_coordstovals(coords_batch)
        inputs = torch.from_numpy(inputs)
        inputs = torch.transpose(inputs, 1, 2).float()
        if self.targets_coordstovals is not None:
            targets = self.targets_coordstovals(coords_batch)
            targets = torch.from_numpy(targets).float()
            return inputs, targets
        else:
            return inputs

    def __len__(self):
        return len(self.coordsbatch_producer)

    def on_epoch_end(self):
        self.coordsbatch_producer.on_epoch_end()


if __name__ == '__main__':
    pass

    dataset = BatchDataset(TF='CTCF', seq_len=1000, is_aug=False)
    print(len(dataset))
    # for inputs, targets in dataset:
    #     print(inputs.shape, targets.shape)
