from data_processing.maru_voice import MaruDataset
from data_processing.urban_sound_8K import UrbanSound8K
from data_processing.dataset import Dataset
import warnings

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')

    maru_basepath = '/Users/gimgibeom/Documents/maru_data/'
    urbansound_basepath = '/Users/gimgibeom/Documents/UrbanSound8K'

    mcv = MaruDataset(maru_basepath, val_dataset_size=200)
    clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()

    us8K = UrbanSound8K(urbansound_basepath, val_dataset_size=200)
    noise_train_filenames, noise_val_filenames = us8K.get_train_val_filenames()

    windowLength = 256
    config = {'windowLength': windowLength,
              'overlap': round(0.25 * windowLength), # 8ms
              'fs': 16000,
              'audio_max_duration': 0.8}

    val_dataset = Dataset(clean_val_filenames, noise_val_filenames, **config)
    val_dataset.create_tf_record(prefix='val', subset_size=200)

    # train_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)
    # train_dataset.create_tf_record(prefix='train', subset_size=4000)
    #
    # ## Create Test Set
    # clean_test_filenames = mcv.get_test_filenames()
    #
    # noise_test_filenames = us8K.get_test_filenames()
    #
    # test_dataset = Dataset(clean_test_filenames, noise_test_filenames, **config)
    # test_dataset.create_tf_record(prefix='test', subset_size=1000, parallel=False)

