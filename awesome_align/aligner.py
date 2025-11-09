from awesome_align.run_align import word_align
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.configuration_bert import BertConfig
from awesome_align.run_align import set_seed, word_align
from pathlib import Path
from argparse import Namespace
from typing import List, Tuple, Optional, Any
from collections import defaultdict
from io import StringIO
import platform
import traceback
import torch

class Aligner:
    def __init__(self,
                  align_layer: int = 8, 
                  extraction: str = 'softmax', 
                  softmax_threeshold: float = 0.001, 
                  output_prob_file: str = None, 
                  output_word_file: str = None,
                  model_name_or_path: str = 'bert-base-multilingual-cased',
                  config_name: str = None,
                  tokenizer_name: str = None,
                  seed: int = 42,
                  batch_size: int = 32,
                  cache_dir: str = None,
                  no_cuda: bool = False,
                  num_workers: int = 4 if not platform.system() == 'Windows' else 0
                  ):
        
        """
        :param int = 8 aligner_layer: Layer for alignment extraction.
        :param string extraction: Must be either softmax or entmax15.
        :param float = 0.001 softmax_threeshold: idk what that is.
        :param string = None output_prob_file: MUST BE AN ABSOLUTE PATH The output probability file.
        :param string = None output_word_file: MUST BE AN ABSOLUTE PATH The output word file. 
        :param string = bert-base-multilingual-cased model_name_or_path: The model checkpoint for weights initialization. Leave None if you want to train a model from scratch
        :param string = None config_name: Optionnal pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.
        :param string = None tokenizer_name: Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.
        :param integer = 42 seed: Random seed for initialization.
        :param integer = 32 batch_size: idk what this is too.
        :param string = None cache_dir: Optional directory to store the pre-trained models downloaded from s3 (instead of the default one).
        :param bool = false no_cuda: Avoid using CUDA when available.
        :param integer = 4 for non windows integer = 0 for windows num_workers: Number of workers for data loading.
        """
        self.args = Namespace(
                num_workers=num_workers,
                device=torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu"),
                batch_size=batch_size,
                align_layer=align_layer,
                extraction=extraction,
                softmax_threshold=softmax_threeshold,
                output_prob_file=output_prob_file,
                output_word_file=output_word_file,
                model_name_or_path=model_name_or_path,
                config_name=config_name,
                tokenizer_name=tokenizer_name,
                seed=seed,
                cache_dir=cache_dir
            )

        self.config_class, self.model_class, self.tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
        self.model, self.tokenizer = self._setup_classes()

        if self.args.extraction not in ['softmax', 'entmax15']:
            raise ValueError('Extraction must either be softmax or entmax15')

        if output_prob_file:
            if not Path(output_prob_file).exists():
                raise FileNotFoundError(rf'File : {output_prob_file} doesn\'t exist. Are you sure it\'s an absolute path ?')
        
        if output_word_file:
            if not Path(output_word_file).exists():
                raise FileNotFoundError(rf'File : {output_word_file} doesn\'t exist. Are you sure it\'s an absolute path ?')

        set_seed(self.args)

    def _load_sentences(self, data_file: Optional[str] = None, string_data: Optional[str] = None) -> Tuple[List[List[str]], List[List[str]]]:
        src_sents = []
        tgt_sents = []

        if data_file:
            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            f = StringIO(string_data)
            lines = f.readlines()
            f.close()

        for line in lines:
            if "|||" not in line:
                continue
            src, tgt = line.strip().split("|||")
            src_sents.append(src.strip().split())
            tgt_sents.append(tgt.strip().split())

        return src_sents, tgt_sents

    def _load_alignments(self, align_file: Optional[str] = None, string_output: Optional[str] = None) -> List[List[Tuple[int, int]]]:
        all_aligns = []

        if align_file:
            with open(align_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            f = StringIO(string_output)
            lines = f.readlines()
            f.close()

        for line in lines:
            pairs = []
            for ij in line.strip().split():
                if '-' not in ij:
                    continue
                i, j = ij.split('-')
                pairs.append((int(i), int(j)))
            all_aligns.append(pairs)

        return all_aligns

    def _build_segments(self, src_sents, tgt_sents, all_aligns) -> list[dict]:
        results = []

        for src_tokens, tgt_tokens, aligns in zip(src_sents, tgt_sents, all_aligns):
            segments = defaultdict(list)

            for i, j in aligns:
                if i < len(src_tokens) and j < len(tgt_tokens):
                    segments[j].append(src_tokens[i])

            sentence_mapping = []
            for j in sorted(segments.keys()):
                src_segment = " ".join(segments[j])
                tgt_segment = tgt_tokens[j]
                sentence_mapping.append({'src_segment': src_segment, 'tgt_segment': tgt_segment})

            results.append(sentence_mapping)

        return results

    def _setup_classes(self) -> tuple[Any, Any, Any]:    
        """Configures the config, model and tokenizer based on the user's arguments."""
        # CONFIG SETUP
        if self.args.config_name:
            config = self.config_class.from_pretrained(self.args.config_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name_or_path:
            config = self.config_class.from_pretrained(self.args.model_name_or_path, cache_dir=self.args.cache_dir)
        else:
            config = self.config_class()

        #TOKENIZER CONFIG
        if self.args.tokenizer_name:
            tokenizer = self.tokenizer_class.from_pretrained(self.args.tokenizer_name, cache_dir=self.args.cache_dir)
        elif self.args.model_name_or_path:
            tokenizer = self.tokenizer_class.from_pretrained(self.args.model_name_or_path, cache_dir=self.args.cache_dir)
        else:
            raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(self.tokenizer_class.__name__)
        )

        #MODEL CONFIG
        if self.args.model_name_or_path:
            model = self.model_class.from_pretrained(
                self.args.model_name_or_path,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=config,
                cache_dir=self.args.cache_dir,
            )
        else:
            model = self.model_class(config=config)

        return model, tokenizer

    def format_output(self):
        src_sents, tgt_sents = self._load_sentences(data_file=self.args.data_file, string_data=self.args.string_data)
        all_aligns = self._load_alignments(align_file=self.args.output_file, string_output=self.results)

        return self._build_segments(src_sents, tgt_sents, all_aligns)

    def align(self, string_data: str = None, data_file: str = None, output_file: str = None, return_output_as_string: bool = None) -> None:
        """
        :param string data_file: Path to your data file. MUST BE ABSOLUTE AND MUST EXIST.
        :param string output_file: Path to your output file. Can be relative and doesn't need to exist.
        :param string string_data: Your data in string instead of a file.
        :param boolean return_output_as_string: Returns the output as a string. Note that it can't be set to true if there is an output file.
        """
        self.args.data_file = data_file
        self.args.output_file = output_file
        self.args.string_data = string_data.strip() if string_data else None
        self.args.return_output_as_string = return_output_as_string

        self.results = None

        if return_output_as_string is None:
            self.args.return_output_as_string = False if output_file else True

        if string_data and data_file:
            raise ValueError('Must either have a data_file or input valid string. Can\'t proccess both.')

        if output_file and return_output_as_string:
            raise ValueError('Return_output_as_string isn\'t supported for now when data is passed as a file. ' \
            'It only works when data is passed as a string. ' \
            'It is planned to be added in the future.')

        if data_file:
            if not Path(data_file).exists():
                raise FileNotFoundError(rf'File : {data_file} doesn\'t exist. Are you sure it\'s an absolute path ?')
            
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            self.results = word_align(self.args, self.model, self.tokenizer)
        except:
            err = traceback.format_exc()
            print(err)
            print('Readjust the number of workers to 0 in order to not make the script crash. Windows doesn\'t support more than 0 workers')

        return self.results
        
