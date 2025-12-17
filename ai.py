class AiCategorizer:
    def predict(self, df: pd.DataFrame, account_store) -> pd.DataFrame:
        """
        returns df with ai_account_number, ai_score in [0,1], ai_side ('soll'/'haben'/'none')
        """
        ...
