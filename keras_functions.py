
def random_crop(img, mask, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx), :]

def foreground_sparse_accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.reshape(y_true, (-1, nb_classes))
    pred_pixels = K.argmax(y_pred, axis=-1)
    true_pixels = K.argmax(y_true, axis=-1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = tf.cast(unpacked[0], tf.bool) | K.greater(K.sum(y_true, axis=-1), 0)
    return K.sum(tf.to_float(~legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(~legal_labels))

def background_sparse_accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.reshape(y_true, (-1, nb_classes))
    pred_pixels = K.argmax(y_pred, axis=-1)
    true_pixels = K.argmax(y_true, axis=-1)
    legal_labels = K.greater(K.sum(y_true, axis=-1), 0)
    return K.sum(tf.to_float(legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(legal_labels))

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.reshape(y_true, (-1, nb_classes))
    legal_labels = K.greater(K.sum(y_true, axis=-1), 0)
    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1),
                                                    K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def segmentation_generator(image_gen, mask_gen):
    while True:
            X = image_gen.next()
            y = mask_gen.next()
            X, y = random_crop(X, y, (256,256))
            y = y.astype('int32')
            y = np.reshape(y, (-1, np.prod(y.shape[1:3]), 1))
            sample_weights = np.ones(y.shape)
            sample_weights[y==255] = 0
            y[y==255]=MAX_LABEL+1
            yield X, y, sample_weights[...,0]

            
from keras.callbacks import *

class CyclicLR(Callback):
    
#     cl = CyclicLR(base_lr=0.0001, max_lr=0.005,
#               step_size=steps, mode = 'cosine', gamma = 0.999,
#               scale_mode='iterations', cycle_mult = 2)
    
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=50., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', cycle_mult = 1):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.init_max_lr = max_lr
        self.step_size = step_size
        self.slope = (base_lr - max_lr)/step_size
        self.mode = mode
        self.gamma = gamma
        self.cycle_mult = cycle_mult
        self.cycle = 0
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(1.2**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'cosine':
                self.scale_fn = lambda x: np.cos(np.pi * x / step_size)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.iterations     = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        self.iterations = 1.
        self.step_size/=(2**self.cycle)
        self.cycle = 0
        if self.mode == 'cosine':
            self.scale_fn = lambda x: np.cos(np.pi * x / self.step_size)
        if self.mode == 'triangular2':
            self.max_lr = self.init_max_lr
            
    def clr(self):
        if (self.iterations % self.step_size)==1:
            self.iterations = 1.
            self.cycle+=1
            self.step_size*=self.cycle_mult
            if self.mode == 'triangular2':
                self.max_lr *= self.scale_fn(self.cycle)
            self.slope = (self.base_lr - self.max_lr)/self.step_size
            if self.mode == 'cosine':
                self.scale_fn = lambda x: np.cos(np.pi * x / self.step_size)
        
        if self.iterations == self.step_size:
            itr = self.iterations
        else:
            itr = self.iterations % self.step_size
        if self.scale_mode == 'cycle':
            return np.maximum(self.base_lr,
                              (self.slope * itr + self.max_lr))
        else:
            A = self.max_lr - self.base_lr
            return (A/2*(self.scale_fn(self.iterations) + 1) + self.base_lr) * self.gamma**(self.clr_iterations)
                
    def on_train_begin(self, logs={}):
        logs = logs or {}
        
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        self.iterations     += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

def calculate_iou(y_preds, labels):
    nb_classes = y_preds.shape[-1]
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = len(labels)
    mean_acc = 0.
    preds = np.argmax(y_preds, axis = -1)
    for i in range(total):
        flat_pred = np.ravel(preds[i])
        flat_label = np.ravel(labels[i])
        acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', i)
            if l==p:
                acc+=1
        acc /= flat_pred.shape[0]
        mean_acc += acc
    mean_acc /= total
    print('mean acc: %f'%mean_acc)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU, mean_acc
