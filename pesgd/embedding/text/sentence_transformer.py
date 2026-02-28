import pandas as pd
from sentence_transformers import SentenceTransformer as ST

from pesgd.embedding import Embedding
from pe.logging import execution_logger
from pe.constant.data import TEXT_DATA_COLUMN_NAME
from pe.constant.data import EMBEDDING_COLUMN_NAME
from pesgd.llm.huggingface.model_name_to_path import SENTENCE_TRANSFORMER_NAME_TO_PATH


class SentenceTransformer(Embedding):
    """Compute the Sentence Transformers embedding of text."""

    def __init__(self, model, batch_size=2000, use_local_model=False):
        """Constructor.

        :param model: The Sentence Transformers model to use
        :type model: str
        :param batch_size: The batch size to use for computing the embedding, defaults to 2000
        :type batch_size: int, optional
        """
        super().__init__()
        self._model_name = model
        if use_local_model:
            self._model_name = SENTENCE_TRANSFORMER_NAME_TO_PATH[self._model_name]
        self._model = ST(self._model_name)
        self._batch_size = batch_size

    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}.{self._model_name}"

    def compute_embedding(self, data):
        """Compute the Sentence Transformers embedding of text.

        :param data: The data object containing the text
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding
        :rtype: :py:class:`pe.data.Data`
        """
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed")
            return data
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)}"
            " samples"
        )
        samples = uncomputed_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        embeddings = self._model.encode(samples, batch_size=self._batch_size)
        # uncomputed_data.data_frame[self.column_name] = pd.Series(
        uncomputed_data.data_frame.loc[:, self.column_name] = pd.Series( # locate columns using its names, if we know about the column index, use .iloc instead.
            list(embeddings), index=uncomputed_data.data_frame.index
        )
        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)
