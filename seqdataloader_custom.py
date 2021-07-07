import gzip
import numpy as np
from collections import namedtuple
from pyfaidx import Fasta

Coordinates = namedtuple("Coordinates",
                         ["chrom", "start", "end", "isplusstrand"])
Coordinates.__new__.__defaults__ = (True,)


def apply_mask(tomask, mask):
    if isinstance(tomask, dict):
        return dict([(key, val[mask]) for key, val in tomask.items()])
    elif isinstance(tomask, list):
        return [x[mask] for x in mask]
    else:
        return [mask]


class KerasSequenceApiCoordsBatchProducer(object):
    """
    Args:
        batch_size (int): note that if you apply some kind of augmentation,
            then this value will end up being half of the actual batch size.
        shuffle_before_epoch (boolean, optional): default False
        seed (int): default 1234; needed if shuffle=True
    """

    def __init__(self, batch_size, shuffle_before_epoch, seed):
        self.coords_list = self._get_coordslist()
        self.batch_size = batch_size
        self.shuffle_before_epoch = shuffle_before_epoch
        self.seed = seed
        if (self.shuffle_before_epoch):
            self.rng = np.random.RandomState(self.seed)
            self._shuffle_coordslist()

    def _get_coordslist(self):
        raise NotImplementedError()

    def _shuffle_coordslist(self):
        self.rng.shuffle(self.coords_list)

    def __getitem__(self, index):
        """
        Args:
            index (:obj:`int`): index of the batch

        Returns:
            :obj:`list`: the coordinates for a complete batch
        """
        return self.coords_list[index * self.batch_size:
                                (index + 1) * self.batch_size]

    def __len__(self):
        """
        Returns:
            The total number of batches to return
        """
        return int(np.ceil(len(self.coords_list) / float(self.batch_size)))

    def on_epoch_end(self):
        """
        Things to be executed after the epoch - like shuffling the coords
        """
        if (self.shuffle_before_epoch):
            self._shuffle_coordslist()


class BedFileObj(object):
    def __init__(self, bed_file, hastitle=False):
        print("Heads up: coordinates in bed file"
              + " are assumed to be on the positive strand;"
              + " if strand in the bed file is improtant to you, please"
              + " add that feature to SimpleCoordsBatchProducer")
        self.bed_file = bed_file
        self.hastitle = hastitle
        self.coords_list = self._read_bed_file()

    def _read_bed_file(self):
        coords_list = []
        for linenum, line in enumerate((gzip.open(self.bed_file) if ".gz"
                                                                    in self.bed_file
        else open(self.bed_file))):
            if (linenum > 0 or self.hastitle == False):
                (chrom, start_str, end_str) = \
                    line.decode("utf-8").rstrip().split("\t")[0:3]
                coords_list.append(Coordinates(chrom=chrom,
                                               start=int(start_str),
                                               end=int(end_str)))
        return coords_list

    def __len__(self):
        return len(self.coords_list)

    def get_strided_subsample(self, offset, stride):
        return self.coords_list[offset::stride]

    def assert_sorted(self):
        prev_entry = self.coords_list[0]
        for entry in self.coords_list[1:]:
            if entry.chrom == prev_entry.chrom:
                assert entry.start >= prev_entry.start, ("Bed file " +
                                                         self.bed_file + " is not sorted; " + str(entry)
                                                         + " follows " + str(prev_entry))
            prev_entry = entry


class DownsampleNegativesCoordsBatchProducer(
    KerasSequenceApiCoordsBatchProducer):

    def __init__(self, pos_bed_file, neg_bed_file,
                 target_proportion_positives, **kwargs):

        print("Reading in positive bed file")
        self.pos_bedfileobj = BedFileObj(bed_file=pos_bed_file)
        print("Got", len(self.pos_bedfileobj.coords_list),
              " coords in positive bed file")
        print("Reading in negative bed file")
        self.neg_bedfileobj = BedFileObj(bed_file=neg_bed_file)
        print("Got", len(self.neg_bedfileobj.coords_list),
              " coords in negative bed file")
        self.neg_bedfileobj.assert_sorted()

        self.target_proportion_positives = target_proportion_positives
        self.subsample_factor = int(np.ceil(
            (len(self.neg_bedfileobj.coords_list)
             * (self.target_proportion_positives /
                (1 - self.target_proportion_positives))) /
            len(self.pos_bedfileobj.coords_list)))
        print("The target proportion of positives of",
              self.target_proportion_positives, "requires the negative set"
              + " to be subsampled by a factor of", self.subsample_factor,
              "which will result in a #neg of",
              int(len(self.neg_bedfileobj.coords_list) / self.subsample_factor))
        self.last_used_offset = -1
        super(DownsampleNegativesCoordsBatchProducer, self).__init__(**kwargs)

    def _shuffle_coordslist(self):
        self.rng.shuffle(self.subsampled_neg_coords)
        self.rng.shuffle(self.pos_coords)
        fracpos = len(self.pos_coords) / (
                len(self.pos_coords) + len(self.subsampled_neg_coords))
        # interleave evenly
        pos_included = 0
        neg_included = 0
        new_coordslist = []
        for i in range(len(self.pos_coords) + len(self.subsampled_neg_coords)):
            if (pos_included < (pos_included + neg_included) * (fracpos)):
                new_coordslist.append(self.pos_coords[pos_included])
                pos_included += 1
            else:
                new_coordslist.append(self.subsampled_neg_coords[neg_included])
                neg_included += 1
        assert pos_included == len(self.pos_coords)
        assert neg_included == len(self.subsampled_neg_coords)
        self.coords_list = new_coordslist

    def _get_coordslist(self):
        self.last_used_offset += 1
        self.last_used_offset = self.last_used_offset % self.subsample_factor
        print("Using an offset of ", self.last_used_offset, " before striding")
        self.last_used_offset = self.last_used_offset % self.subsample_factor
        subsampled_neg_coords = self.neg_bedfileobj.get_strided_subsample(
            offset=self.last_used_offset,
            stride=self.subsample_factor)
        pos_coords = self.pos_bedfileobj.coords_list
        self.subsampled_neg_coords = subsampled_neg_coords
        self.pos_coords = pos_coords
        return pos_coords + subsampled_neg_coords

    def on_epoch_end(self):
        # get negative set with potentially different stride
        self.coords_list = self._get_coordslist()
        # perform shuffling as needed
        super(DownsampleNegativesCoordsBatchProducer, self).on_epoch_end()


class SimpleCoordsBatchProducer(KerasSequenceApiCoordsBatchProducer):
    """
    Args:
        bed_file (string): file with the bed coordinates.
            Assumes coordinates are on the positive strand.
        coord_batch_transformer (AbstracCoordBatchTransformer): does things
            like revcomp and random jitter
    """

    def __init__(self, bed_file,
                 hastitle=False,
                 coord_batch_transformer=None,
                 **kwargs):
        self.bed_file = BedFileObj(bed_file=bed_file, hastitle=hastitle)
        if (coord_batch_transformer is not None):
            raise DeprecationWarning(
                "Moving forward, coords_batch_transformer should be"
                + " specified as an argument to KerasBatchGenerator"
                + ", not as an arugment to the CoordsBatchProducer."
                + " This is to allow different CoordsBatchProducer"
                + " implementations to be used with the same"
                + " coords_batch_transformer code.")
        self.coord_batch_transformer = coord_batch_transformer
        super(SimpleCoordsBatchProducer, self).__init__(**kwargs)

    def _get_coordslist(self):
        return [x for x in self.bed_file.coords_list]

    def __getitem__(self, index):
        orig_batch = self.coords_list[index * self.batch_size:
                                      (index + 1) * self.batch_size]
        if (self.coord_batch_transformer is not None):
            return self.coord_batch_transformer(orig_batch)
        else:
            return orig_batch


def get_new_coors_around_center(coors, center_size_to_use):
    new_coors = []
    for coor in coors:
        coor_center = int(0.5 * (coor.start + coor.end))
        left_flank = int(0.5 * center_size_to_use)
        right_flank = center_size_to_use - left_flank
        new_start = coor_center - left_flank
        new_end = coor_center + right_flank
        new_coors.append(Coordinates(chrom=coor.chrom,
                                     start=new_start, end=new_end,
                                     isplusstrand=coor.isplusstrand))
    return new_coors


class CoordsToVals(object):

    def __call__(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates`):
        Returns:
            numpy ndarray OR list of ndarrays OR a dict of mode_name->ndarray.
              Returns a list of ndarrays if returning multiple modes.
              Alternatively, returns a dict where key is the mode name
              and the value is the ndarray for the mode.
        """
        raise NotImplementedError()


class CoordsToValsJoiner(CoordsToVals):

    def __init__(self, coordstovals_list):
        """
        Joins batches returned by other CoordsToVals objects
        Args:
            coorstovals_list (:obj:`list` of :obj:`CoordsToVals`): List of
                CoordsToVals whose values to combine
        """
        self.coordstovals_list = coordstovals_list

    def __call__(self, coors):
        batch_to_return = None
        for idx, coordstovals_obj in enumerate(self.coordstovals_list):
            the_batch = coordstovals_obj(coors=coors)
            assert the_batch is not None
            if isinstance(the_batch, dict):
                assert ((batch_to_return is None) or
                        (isinstance(batch_to_return, dict))), (
                        "coordstovals object at idx" + str(idx)
                        + " returned a dict, but previous coordstovals"
                        + " objects had a return type incompatible with this")
                if (batch_to_return is None):
                    batch_to_return = {}
                for key in the_batch:
                    assert key not in batch_to_return, (
                            "coordstovals object at idx" + str(idx)
                            + " returned a dict with a key of " + key
                            + ", which collides with a pre-existing key returned by"
                            + " another coordstovals object")
                batch_to_return.update(the_batch)
            else:
                assert ((batch_to_return is None) or
                        (isinstance(batch_to_return, list))), (
                        "coordstovals object at idx" + str(idx)
                        + " returned a type incompatible with dict, but previous"
                        + " coordstovals objects had a return type of dict")
                if (isinstance(the_batch, list) == False):
                    the_batch = [the_batch]
                if (batch_to_return is None):
                    batch_to_return = []
                batch_to_return.extend(the_batch)
        if (batch_to_return is None):
            batch_to_return = []
        return batch_to_return


class AbstractSingleNdarrayCoordsToVals(CoordsToVals):

    def __init__(self, mode_name=None):
        """
        Args:
            mode_name (:obj:`str`, optional): default None. If None, then
                the return of __call__ will be a numpy ndarray. Otherwise, it
                will be a dictionary with a key of mode_name and a value being
                the numpy ndarray.
        """
        self.mode_name = mode_name

    def _get_ndarray(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates):

        Returns:
            numpy ndarray
        """
        raise NotImplementedError()

    def __call__(self, coors):
        ndarray = self._get_ndarray(coors)
        if (self.mode_name is None):
            return ndarray
        else:
            return {self.mode_name: ndarray}


# Coords -> 0/1
class SimpleLookup(AbstractSingleNdarrayCoordsToVals):

    def __init__(self, lookup_file,
                 transformation=None,
                 default_returnval=0.0, **kwargs):
        super(SimpleLookup, self).__init__(**kwargs)
        self.lookup_file = lookup_file
        self.transformation = transformation
        self.default_returnval = default_returnval
        self.lookup = {}
        self.num_labels = None
        for line in (gzip.open(self.lookup_file) if ".gz"
                                                    in self.lookup_file else open(self.lookup_file)):
            (chrom, start_str, end_str, *labels) = \
                line.decode("utf-8").rstrip().split("\t")
            coord = Coordinates(chrom=chrom,
                                start=int(start_str),
                                end=int(end_str))
            labels = [(self.transformation(float(x))
                       if self.transformation is not None else float(x))
                      for x in labels]
            self.lookup[(coord.chrom, coord.start, coord.end)] = labels
            if (self.num_labels is None):
                self.num_labels = len(labels)
            else:
                assert len(labels) == self.num_labels, (
                        "Unequal label lengths; " + str(len(labels), self.num_labels))

    def _get_ndarray(self, coors):
        to_return = []
        for coor in coors:
            if (coor.chrom, coor.start, coor.end) not in self.lookup:
                to_return.append(np.ones(self.num_labels)
                                 * self.default_returnval)
            else:
                to_return.append(
                    self.lookup[(coor.chrom, coor.start, coor.end)])
        return np.array(to_return)


ltrdict = {
    'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1],
    'n': [0, 0, 0, 0], 'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}


def onehot_encoder(seq):
    return np.array([ltrdict.get(x, [0, 0, 0, 0]) for x in seq])


# Coords -> one_hot(A,C,T,G
class PyfaidxCoordsToVals(AbstractSingleNdarrayCoordsToVals):

    def __init__(self, genome_fasta_path, center_size_to_use=None, **kwargs):
        """
        Args:
            genome_fasta_path (:obj:`str`): path to the genome .fa file
            **kwargs: arguments for :obj:`AbstractSingleNdarrayCoordsToVals`
        """
        super(PyfaidxCoordsToVals, self).__init__(**kwargs)
        self.center_size_to_use = center_size_to_use
        self.genome_fasta = genome_fasta_path

    def _get_ndarray(self, coors):
        """
        Args:
            coors (:obj:`list` of :obj:`Coordinates): if
                center_size_to_use is not specified, all the
                coordinates must be of the same length

        Returns:
            numpy ndarray of dims (nexamples x width x 4)
        """
        genome_object = Fasta(self.genome_fasta)
        seqs = []
        for coor in coors:
            if (self.center_size_to_use is not None):
                the_center = int((coor.start + coor.end) * 0.5)
                if (coor.chrom in genome_object):
                    seqs.append(genome_object[coor.chrom][
                                the_center - int(0.5 * self.center_size_to_use):
                                the_center + (self.center_size_to_use
                                              - int(0.5 * self.center_size_to_use))])
                else:
                    print(coor.chrom + " not in " + self.genome_fasta)
            else:
                if (coor.chrom in genome_object):
                    seqs.append(genome_object[coor.chrom][coor.start:coor.end])
                else:
                    print(coor.chrom + " not in " + self.genome_fasta)
        genome_object.close()

        onehot_seqs = []
        for seq, coor in zip(seqs, coors):
            onehot = onehot_encoder(seq=seq.seq)
            if (coor.isplusstrand == False):
                onehot = onehot[::-1, ::-1]
            onehot_seqs.append(onehot)
        lengths = set([len(x) for x in onehot_seqs])
        if (len(lengths) > 0):
            assert len(lengths) == 1, ("All the sequences must be of the same"
                                       + "lengths, but lengths are " + str(lengths))
        return np.array(onehot_seqs)


