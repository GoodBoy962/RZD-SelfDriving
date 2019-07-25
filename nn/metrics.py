from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    y_pred = K.round(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
    y_pred = K.round(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum( y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f*y_pred_f)
    fp = K.sum((1-y_true_f)*y_pred_f)
    fn = K.sum(y_true_f*(1-y_pred_f))

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    return 2*p*r / (p+r+K.epsilon())

def kulch_coef(y_true, y_pred, smooth=1):
    y_pred = K.round(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - 2*intersection + smooth)

def kulch_coef_loss(y_true, y_pred):
    return 1-kulch_coef(y_true, y_pred)

def bce_kulch_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + kulch_coef_loss(y_true, y_pred)