Design an Efficient Deep Learning Data Loading Module

Data loading is an important part of the machine learning system, especially when the data is huge and doesn't fit into memory.  The general design goal for a data loading module is to achieve more efficient data loading, with less effort spent on data preparation, and a clean and flexible interface.

This topic is organized as follows: 


- "Design Insight" provides insights into and guidelines for our data loading design.
- "Formatting Data" introduces our solution using dmlc-core's binary recordio implementation.
- "Loading and Preprocessing Data" explains our method of hiding I/O cost by using the Threadediter provided by dmlc-core.
- "Creating an MXNet I/O Interface with Python" shows a simple way to construct an MXNet data iterator in a few lines of Python code. 
- "Roadmap" discusses plans for documenting data I/O solutions for other types of applications.

We cover the following key requirements in detail within these topics:

- Maintaining small file size
- Allowing parallel (distributed) packing of data
- Loading data fast and online augmentation
- Allowing fast reading of arbitrary parts in a distributed setting

## Design Insight
I/O design involves data preparation and data loading. Data preparation occurs offline, and data loading influences online performance. This section describes our I/O design for both phases.

### Preparing Data 
Data preparation involves packing the data into a certain format for later processing. When the data is huge, i.e, a full ImageNet, this process can be time consuming. We recommend the following:

- Pack the dataset into a few files. A dataset might contain millions of data instances. Packed data can be distributed easily among devices.
- Pack once. There's no need to repack when a running setting has been changed (usually, the number of running machines).
- Process packing in parallel to save time.
- Provide easy access to arbitrary parts. This is crucial for distributed machine learning when data parallelism is introduced. When you've packed the data into several physical data files, this can get tricky. Ideally, no matter how many physical data files there are, the packed data can be logically partitioned into an arbitrary number of partitions. For example, we pack 1000 images into 4 physical files, each containing 250 images. We use 10 devices to train DNN, and we should be able to load approximately 100 images per device. Some devices might need images from different physical files.

### Loading Data
Data loading involves loading the packed data into RAM. We want to load the data as quickly as possible. We recommend the following:

- Implement continuous reading. This prevents arbitrary reading from disk.
- Reduce the bytes to be loaded. Achieve this by storing the data instance compactly, e.g., save the image in JPEG format.
- Load and train in different threads. This reduces loading time.
- Avoid using RAM. If the packed file is huge, don't load the whole file into RAM.

## Formatting Data 

Because training deep neural networks always involves processing a huge amount of data, the data format must support efficient and convenient processing.

To achieve the goals described in "Data Insight," we need to pack binary data into a splittable format. In MXNet, we use the binary recordio format implemented in dmlc-core as the basic data saving format.

### Using Binary Records

![baserecordio](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/baserecordio.jpg)

In binary recordio, each data instance is stored as a record. `kMagic` is a magic number indicating the start of a record. `Irecord` encodes length and continues flat. 
In `irecord`:
  
- cflag == 0: This is a complete record. 
- cflag == 1: This is the start of a multiple-rec. 
- cflag == 2: This is the middle of a multiple-rec. 
- cflag == 3: This is the end of a multiple-rec. 

`Data` is the space in which to save data. `Pad` is a padding space to make records align to 4 bytes.

After packing, each file contains multiple records. Loading can continue. This avoids the poor performance of randomly reading from disk.

One great advantage of storing data as records is that records can vary in length. If a compact algorithm is available for a certain kind of data, this allows us to save data more compactly. For example, if you use JPEG format to save image data, the packed data will be much smaller than if it were stored in RGB format. Using the ImageNet_1K dataset as an example, if you store the data with 3 * 256 * 256 raw rgb values, the dataset would occupy more than 200 GB. After compacting it into JPEG format, it occupies only about 35 GB of disk space. This could greatly reduce the cost of reading the disk.

Here's an example of Image binary recordio:
![baserecordio](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/ImageRecordIO.jpg)
First, we resize the image to 256 * 256, then save it in JPEG format. We save the header, which indicates the index and label for the image, to construct the `Data` field of a record. Then we pack several images into a file.

### Accessing Arbitrary Parts Of Data

One goal for data loading is that the packed data can be logically sliced into an arbitrary number of partitions, no matter how many physical packed data files there are.

Because binary recordio can easily locate the start and end of a record by using the magic number, we can achieve this goal by using the InputSplit functionality provided by dmlc-core.

InputSplit takes the following parameters:

- FileSystem *filesys. dmlc-core encapsulates the I/O operations for different file systems, such as HDFS, Amazon S3, and local. You don't need to worry about the difference between file systems anymore.
- Char *uri. The uri of files. If you pack the data into several physical parts, this could be a list of files. Separate file URIs with a ';'.
- Unsigned nsplit. The number of logical splits. nsplit can differ from the number of physical file parts.
- Unsigned rank. Which split to load in this process.

The following figures demonstrate the splitting process.

- The file size of each physical part. Each file contains several records.

![beforepartition](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/beforepartition.jpg)

- Approximately partition according to file size. The boundary of a part might be located in the middle of a record.

![approxipartition](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/approximatepartition.jpg)

-  Finish partitioning by seeking the beginning of records to avoid incomplete records.

![afterpartition](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/afterpartition.jpg)

After conducting these operations, you have identified the records that belong to each part and the physical data files needed by each logical part. InputSplit greatly reduces the difficulty of implementing data parallelism, where each process reads only part of the data.

Because logical partitioning doesn't rely on the number of physical data files, you easily can process huge datasets like ImageNet_22K in parallel, as shown in the following figure. You don't need to consider distributed loading issues during preparation. Just choose the most efficient number of physical files according to the size of the dataset and your computing resources.
![parallelprepare](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/parallelprepare.jpg)

## Loading and Preprocessing Data

When loading and preprocessing speed doesn't keep up with training or evaluation speed, I/O becomes a bottleneck for the whole system. In this topic, we discuss tricks we used to achieve ultimate efficiency for loading and preprocessing data packed in binary recordio format. In our experience with ImageNet, we've achieved I/O speed of `3000` images/sec with a standard HDD.

### Loading and Preprocessing on the Fly

When training deep neural networks, sometimes you can load and preprocess the data only when training. This happens for the following reasons:

- The size of the dataset exceeds the available RAM, so you can't load data in advance.
- If you introduce randomness in training, the preprocessing pipeline might produce different output for the same data at different epochs.

To achieve ultimate efficiency, use the multi-thread technique. Using ImageNet training as an example, after loading a bunch of image records, start multiple threads to perform image decoding and image augmentation, as illustrated in the following figure:
![process](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/process.jpg)

### Reducing I/O Cost Using Threadediter

One way to reduce I/O cost is to prefetch the data for  the next batch on a standalone thread, while the main thread is performing feed-forward and backward. To support more complicated training schemas, MXNet provides a more general I/O processing pipeline using threadediter. Threadediter is provided by dmlc-core.

The key to threadediter is to start a standalone thread  that acts like a data provider, while the main thread acts like a data consumer, as illustrated in the following figure.

Threadediter will maintain a buffer of a certain size and automatically fill the buffer if it's not full. After the data consumer finishes consuming part of the data in the buffer, threadediter reuses the space to save the next part of data.
![threadediter](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/io/threadediter.png)

## Creating an MXNet I/O Interface with Python
We use NumPy to make the I/O object an iterator. That allows us to easily access the data by using a for-loop or calling the next() function. In MXNet, defining a data iterator is very similar to defining a symbolic operator.

The following example shows how to create a CIFAR data iterator:

    ```python
    dataiter = mx.io.ImageRecordIter(
        # Dataset Parameter, indicating the data file, please check the data is already there
        path_imgrec="data/cifar/train.rec",
        # Dataset Parameter, indicating the image size after preprocessing
        data_shape=(3,28,28),
        # Batch Parameter, tells how many images in a batch
        batch_size=100,
        # Augmentation Parameter, when offers mean_img, each image will subtract the mean value at each pixel
        mean_img="data/cifar/cifar10_mean.bin",
        # Augmentation Parameter, randomly crop a patch of the data_shape from the original image
        rand_crop=True,
        # Augmentation Parameter, randomly mirror the image horizontally
        rand_mirror=True,
        # Augmentation Parameter, randomly shuffle the data
        shuffle=False,
        # Backend Parameter, preprocessing thread number
        preprocess_threads=4,
        # Backend Parameter, prefetch buffer size
        prefetch_buffer=1)
     ```

Typically, to create a data iterator, you need to provide five kinds of parameters:

* `Dataset Param` provides basic information about the dataset, e.g., file path, input shape.
* `Batch Param` provides the information for forming a batch, e.g., batch size.
* `Augmentation Param` tells which augmentation operations (e.g., crop, mirror) should be performed on an input image.
* `Backend Param` controls the behavior of the back-end threads to reduce the cost of  data loading.
* `Auxiliary Param` provides options to help checking and debugging.

Usually, you *must* provide `Dataset Param` and `Batch Param`; otherwise, the data batch can't be created. Provide other parameters based on algorithm and performance needs, or use the default values.

Ideally, you should separate the MX data I/O into modules, two of which might be useful to expose to users:


- Efficient prefetcher allows the user to write a data loader that reads a customized binary format, and automatically supports the multi-thread prefetcher.

- Data transformer enables random cropping of images, mirroring, etc., and allows users to plug in their own customized transformers (for example, to add a specific kind of coherent random noise to data, etc.).

## Roadmap

We will provide details on data I/O for image segmentation, object localization, and speech recognition applications when such applications are running on MXNet.

## Next Steps

* [Survey of the RNN Interface](http://mxnet.io/architecture/rnn_interface.html)