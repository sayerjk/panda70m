""""writer module handle writing the videos to disk"""

import json
import os
from pprint import pprint

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds
import logging
import csv
from datetime import datetime


class BufferedParquetWriter:
    """Write samples to parquet files incrementally with a buffer"""

    def __init__(self, output_file, schema, buffer_size=100):
        self.buffer_size = buffer_size
        self.schema = schema
        self._initiatlize_buffer()
        fs, output_path = fsspec.core.url_to_fs(output_file)

        self.output_fd = fs.open(output_path, "wb")
        self.parquet_writer = pq.ParquetWriter(self.output_fd, schema)

    def _initiatlize_buffer(self):
        self.current_buffer_size = 0
        self.buffer = {k: [] for k in self.schema.names}

    def _add_sample_to_buffer(self, sample):
        for k in self.schema.names:
            self.buffer[k].append(sample[k])
        self.current_buffer_size += 1

    def write(self, sample):
        if self.current_buffer_size >= self.buffer_size:
            self.flush()
        self._add_sample_to_buffer(sample)

    def flush(self):
        """Write the buffer to disk"""
        if self.current_buffer_size == 0:
            return

        df = pa.Table.from_pydict(self.buffer, self.schema)
        self.parquet_writer.write_table(df)
        self._initiatlize_buffer()

    def close(self):
        self.flush()
        if self.parquet_writer is not None:
            self.parquet_writer.close()
            self.parquet_writer = None
            self.output_fd.close()


class ParquetSampleWriter:
    """ParquetSampleWriter is a video+caption writer to parquet"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_formats,
    ):
        self.oom_shard_count = oom_shard_count
        for fmt in encode_formats.values():
            schema = schema.append(pa.field(fmt, pa.binary()))
        shard_name = (
            shard_id
            if isinstance(shard_id, str)
            else "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
                shard_id=shard_id, oom_shard_count=oom_shard_count
            )
        )
        output_file = f"{output_folder}/{shard_name}.parquet"
        self.buffered_parquet_writer = BufferedParquetWriter(output_file, schema, 100)
        self.save_caption = save_caption
        self.encode_formats = encode_formats

    def write(self, streams, key, caption, meta):
        """Keep sample in memory then write to disk when close() is called"""
        sample = {"key": key}
        for modality, stream in streams.items():
            ext = self.encode_formats[modality] if modality in self.encode_formats else modality
            sample[ext] = stream

        if self.save_caption:
            sample["txt"] = str(caption) if caption is not None else ""
        sample.update(meta)

        self.buffered_parquet_writer.write(sample)

    def close(self):
        self.buffered_parquet_writer.close()


class WebDatasetSampleWriter:
    """WebDatasetSampleWriter is a video+caption writer to webdataset"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_formats,
    ):
        self.oom_shard_count = oom_shard_count
        shard_name = (
            shard_id
            if isinstance(shard_id, str)
            else "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
                shard_id=shard_id, oom_shard_count=oom_shard_count
            )
        )
        self.shard_id = shard_id
        fs, output_path = fsspec.core.url_to_fs(output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_formats = encode_formats

    def write(self, streams, key, caption, meta):
        """write sample to tars"""
        sample = {"__key__": key}
        for modality, stream in streams.items():
            ext = self.encode_formats[modality] if modality in self.encode_formats else modality
            sample[ext] = stream

        if self.save_caption:
            sample["txt"] = str(caption) if caption is not None else ""
        # some meta data may not be JSON serializable
        for k, v in meta.items():
            if isinstance(v, np.ndarray):
                meta[k] = v.tolist()
        sample["json"] = json.dumps(meta, indent=4)

        self.tarwriter.write(sample)
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tarwriter.close()
        self.tar_fd.close()


class TFRecordSampleWriter:
    """TFRecordSampleWriter is a video+caption writer to TFRecord"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_formats,
    ):
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow_io as _  # pylint: disable=import-outside-toplevel
            from tensorflow.python.lib.io.tf_record import (  # pylint: disable=import-outside-toplevel
                TFRecordWriter,
            )  # pylint: disable=import-outside-toplevel
            from tensorflow.python.training.training import (  # pylint: disable=import-outside-toplevel
                BytesList,
                Example,
                Feature,
                Features,
                FloatList,
                Int64List,
            )

            self._BytesList = BytesList  # pylint: disable=invalid-name
            self._Int64List = Int64List  # pylint: disable=invalid-name
            self._FloatList = FloatList  # pylint: disable=invalid-name
            self._Example = Example  # pylint: disable=invalid-name
            self._Features = Features  # pylint: disable=invalid-name
            self._Feature = Feature  # pylint: disable=invalid-name
        except ImportError as e:
            raise ModuleNotFoundError(
                "tfrecords require tensorflow and tensorflow_io to be installed."
                "Run `pip install tensorflow tensorflow_io`."
            ) from e

        self.oom_shard_count = oom_shard_count
        shard_name = (
            shard_id
            if isinstance(shard_id, str)
            else "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
                shard_id=shard_id, oom_shard_count=oom_shard_count
            )
        )
        self.shard_id = shard_id
        self.tf_writer = TFRecordWriter(f"{output_folder}/{shard_name}.tfrecord")
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_formats = encode_formats

    def write(self, streams, key, caption, meta):
        """Write a sample using tfrecord writer"""
        sample = {"key": self._bytes_feature(key.encode())}
        for modality, stream in streams.items():
            ext = self.encode_formats[modality] if modality in self.encode_formats else modality
            sample[ext] = self._bytes_feature(stream)

        if self.save_caption:
            sample["txt"] = self._bytes_feature(str(caption) if caption is not None else "")
        for k, v in meta.items():
            sample[k] = self._feature(v)

        tf_example = self._Example(features=self._Features(feature=sample))
        self.tf_writer.write(tf_example.SerializeToString())
        self.buffered_parquet_writer.write(meta)

    def close(self):
        self.buffered_parquet_writer.close()
        self.tf_writer.close()

    def _feature(self, value):
        if isinstance(value, list):
            return self._list_feature(value)
        elif isinstance(value, int):
            return self._int64_feature(value)
        elif isinstance(value, float):
            return self._float_feature(value)
        else:
            return self._bytes_feature(value)

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if value is None:
            value = ""
        if isinstance(value, str):
            value = value.encode()
        return self._Feature(bytes_list=self._BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return self._Feature(float_list=self._FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return self._Feature(int64_list=self._Int64List(value=[value]))

    def _list_feature(self, value):
        """Returns an list of int64_list, float_list, bytes_list."""
        if isinstance(value[0], int):
            return self._Feature(int64_list=self._Int64List(value=value))
        elif isinstance(value[0], float):
            return self._Feature(float_list=self._FloatList(value=value))
        else:
            for i, bytes_feature in enumerate(value):
                if bytes_feature is None:
                    value[i] = ""
                if isinstance(bytes_feature, str):
                    value[i] = bytes_feature.encode()
            return self._Feature(bytes_list=self._BytesList(value=value))


class FilesSampleWriter:
    """FilesSampleWriter is a caption+video writer to files"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_formats,
    ):
        self.oom_shard_count = oom_shard_count
        shard_name = (
            shard_id
            if isinstance(shard_id, str)
            else "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
                shard_id=shard_id, oom_shard_count=oom_shard_count
            )
        )
        self.shard_id = shard_id
        self.fs, self.subfolder = fsspec.core.url_to_fs(f"{output_folder}/{shard_name}")
        if not self.fs.exists(self.subfolder):
            self.fs.mkdir(self.subfolder)
        self.save_caption = save_caption
        self.buffered_parquet_writer = BufferedParquetWriter(output_folder + "/" + shard_name + ".parquet", schema, 100)
        self.encode_formats = encode_formats
        
        # Create a CSV log file for this shard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_log_path = f"{output_folder}/processed_videos_{timestamp}.csv"
        
        logging.info(f"Creating CSV log file at: {self.csv_log_path}")
        
        # Create CSV file with headers
        self.csv_headers = [
            'processing_time',
            'shard_id',
            'video_id',
            'url',
            'timestamp',
            'caption',
            'matching_score',
            'desirable_filtering',
            'status',
            'clip_index'
        ]
        
        # Create or append to CSV file
        try:
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                if f.tell() == 0:  # Only write headers if file is empty
                    writer.writeheader()
                    logging.info("CSV headers written successfully")
        except Exception as e:
            logging.error(f"Error creating CSV file: {e}")

    def write(self, streams, key, caption, meta):
        """Write sample to disk"""
        # First check if we should save
        should_save = self._should_save_video(meta, key)
        
        # Log the processing regardless of save decision
        self._log_to_csv(key, caption, meta, should_save)
        
        if not should_save:
            return
            
        print(f"\nProcessing video {key}")
        # print(f"Metadata received: {meta}")
        
        for modality, stream in streams.items():
            ext = self.encode_formats[modality] if modality in self.encode_formats else modality
            filename = f"{self.subfolder}/{key}.{ext}"
            with self.fs.open(filename, "wb") as f:
                f.write(stream)

        if self.save_caption:
            caption = str(caption) if caption is not None else ""
            caption_filename = f"{self.subfolder}/{key}.txt"
            with self.fs.open(caption_filename, "w") as f:
                f.write(str(caption))

        # some meta data may not be JSON serializable
        for k, v in meta.items():
            if isinstance(v, np.ndarray):
                meta[k] = v.tolist()
        j = json.dumps(meta, indent=4)
        meta_filename = f"{self.subfolder}/{key}.json"
        with self.fs.open(meta_filename, "w") as f:
            f.write(j)

        self.buffered_parquet_writer.write(meta)

    def _log_to_csv(self, key, caption, meta, was_saved):
        """Log processing details to CSV"""
        try:
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                
                # Prepare the log entry
                log_entry = {
                    'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'shard_id': self.shard_id,
                    'video_id': meta.get('videoID', ''),
                    'url': meta.get('url', ''),
                    'timestamp': meta.get('timestamp', ''),
                    'caption': caption if isinstance(caption, str) else str(caption),
                    'matching_score': meta.get('matching_score', ''),
                    'desirable_filtering': meta.get('desirable_filtering', ''),
                    'status': 'SAVED' if was_saved else 'SKIPPED',
                    'clip_index': key
                }
                
                writer.writerow(log_entry)
                logging.debug(f"Wrote entry to CSV for video {key}")
                
        except Exception as e:
            logging.error(f"Error writing to CSV log for video [{key}]: {e}")
            logging.error(f"Attempted to write to: {self.csv_log_path}")

    def _should_save_video(self, meta, key):
        """Check if video should be saved based on metadata"""
        try:
            filtering_value = meta.get('desirable_filtering')
            video_id = meta.get('videoID', 'unknown')
            
            if not isinstance(filtering_value, list):
                filtering_value = [filtering_value]
                
            # should_save = any(val == 'desirable' for val in filtering_value)
            should_save = True
            
            status = 'SAVING' if should_save else 'SKIPPING'
            logging.info(f"Video [index: {key}, id: {video_id}]: {status}")
            logging.debug(f"Filtering values for [{key}]: {filtering_value}")
            
            return should_save
            
        except Exception as e:
            logging.error(f"Error checking metadata for video [{key}]: {e}")
            return False

    def close(self):
        self.buffered_parquet_writer.close()


class DummySampleWriter:
    """Does not write"""

    def __init__(
        self,
        shard_id,
        output_folder,
        save_caption,
        oom_shard_count,
        schema,
        encode_formats,
    ):
        pass

    def write(self, streams, key, caption, meta):
        pass

    def close(self):
        pass
