import os
import dspy
from dotenv import load_dotenv
import dspy.teleprompt
from tnqeet.data import fewshot_val_dataset
from tnqeet import remove_dots
from tnqeet import constants

load_dotenv()


class ArabicDottingSignature(dspy.Signature):
    dotless_text = dspy.InputField(desc="Arabic text without dots (Rasm)")
    dotted_text = dspy.OutputField(desc="Text with proper dots added")


class DetailedArabicDotingSignature(dspy.Signature):
    f"""Arabic text dottization: restore dots to undotted Arabic text (Rasm).
    Context: Arabic text without dots (Rasm) uses simplified letter forms where multiple
    letters share the same basic shape. The dottization process restores the original
    dotted letters based on context and meaning.

    Key letter mappings (undotted → possible dotted forms):
    - {constants.BAA_RASM} (BAA_RASM) → ب ت ث ن
    - {constants.JEEM_RASM} (JEEM_RASM) → ج ح خ
    - {constants.DAL_RASM} (DAL_RASM) → د ذ
    - {constants.RAA_RASM} (RAA_RASM) → ر ز
    - {constants.SEEN_RASM} (SEEN_RASM) → س ش
    - {constants.SAAD_RASM} (SAAD_RASM) → ص ض
    - {constants.TAA_RASM} (TAA_RASM) → ط ظ
    - {constants.AIN_RASM} (AIN_RASM) → ع غ
    - {constants.FAA_RASM} (FAA_RASM) → ف
    - {constants.QAF_RASM} (QAF_RASM) → ق
    - {constants.YAA_RASM} (YAA_RASM) → ي ى ئ
    - {constants.WAW_RASM} (WAW_RASM) → و ؤ
    - {constants.HAA_RASM} (HAA_RASM) → ه ة
    - {constants.NOON_RASM} (NOON_RASM) → ن
    - {constants.KAF_RASM} (KAF_RASM) → ك
    - {constants.LAM_RASM} (LAM_RASM) → ل
    - {constants.MEEM_RASM} (MEEM_RASM) → م
    - {constants.HAMZA_RASM} (HAMZA_RASM) → ء
    - {constants.ALEF_RASM} (ALEF_RASM) → ا أ إ آ

    Note: This version uses direct character-to-character mapping without positional
    preprocessing. Each undotted character should be restored to its most contextually
    appropriate dotted form."""

    dotless_text = dspy.InputField(
        desc="Arabic text without dots (Rasm) - simplified letter forms using direct character mapping"
    )
    dotted_text = dspy.OutputField(
        desc="Properly dotted Arabic text with correct diacritical marks restored based on context and meaning"
    )


class OpenRouterArabicDotter:
    def __init__(
        self,
        api_key=None,
        model="anthropic/claude-sonnet-4",
        dspy_cache=False,
        signature=ArabicDottingSignature,
        num_fewshot=0,
        max_tokens=None,
        fewshot_dataset=None,
        use_openrouter_model=True,
    ):
        if not use_openrouter_model:
            if model.startswith("openai/"):
                api_key = api_key or os.getenv("OPENAI_API_KEY")
            elif model.startswith("anthropic/"):
                api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            else:
                raise ValueError(f"Unsupported model: {model}")
        else:
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key required")

        self.lm = dspy.LM(
            model=model,
            api_key=api_key,
            temperature=0.01,
            model_type="chat",
            cache=dspy_cache,
            max_tokens=max_tokens,  # type: ignore
            api_base="https://openrouter.ai/api/v1" if use_openrouter_model else None,
        )
        dspy.configure(lm=self.lm)

        # Prepare few-shot examples if requested
        fewshot_dataset = fewshot_dataset or fewshot_val_dataset
        self.dotter = dspy.Predict(signature=signature)
        if num_fewshot > 0:
            examples = [
                dspy.Example(
                    dotless_text=remove_dots(sample["text"]),  # type: ignore
                    dotted_text=sample["text"],  # type: ignore
                ).with_inputs("dotless_text")
                for sample in fewshot_dataset
            ]
            fewshot_optimizer = dspy.teleprompt.LabeledFewShot(k=num_fewshot)
            self.dotter = fewshot_optimizer.compile(
                self.dotter,
                sample=False,
                trainset=examples,
            )

    def restore_dots(self, dotless_text):
        prediction = self.dotter(dotless_text=dotless_text)
        return prediction.dotted_text
