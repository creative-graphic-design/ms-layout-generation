from layoutformer_pp.data import PubLayNetDataset, RicoDataset


def test_label_index_roundtrip_rico():
    label2index = RicoDataset.label2index(RicoDataset.labels)
    index2label = RicoDataset.index2label(RicoDataset.labels)

    assert len(label2index) == len(RicoDataset.labels)
    first_label = RicoDataset.labels[0]
    assert index2label[label2index[first_label]] == first_label


def test_label_index_roundtrip_publaynet():
    label2index = PubLayNetDataset.label2index(PubLayNetDataset.labels)
    index2label = PubLayNetDataset.index2label(PubLayNetDataset.labels)

    assert len(label2index) == len(PubLayNetDataset.labels)
    first_label = PubLayNetDataset.labels[0]
    assert index2label[label2index[first_label]] == first_label


def test_dataset_colors_have_expected_length(data_root):
    dataset = RicoDataset(
        root=str(data_root),
        split="val",
        max_num_elements=20,
        online_process=True,
    )
    colors = dataset.colors
    assert len(colors) == len(dataset.label_set) + 1
