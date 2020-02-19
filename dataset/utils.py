from sklearn.preprocessing import MultiLabelBinarizer

def multiLabel_binarizer(cfg, label):

    mlb = MultiLabelBinarizer(classes=range(0, cfg.MODEL.NUM_CLASSES))

    return mlb.fit_transform([label])[0]