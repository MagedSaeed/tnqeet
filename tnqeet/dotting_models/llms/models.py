import os
import json
from importlib import resources

import dspy
from dotenv import load_dotenv
import dspy.teleprompt
from tnqeet import remove_dots
from tnqeet import constants

load_dotenv()


def _load_bundled_fewshot_examples():
    """Load the bundled fewshot examples (dicts with a ``"text"`` field).

    Reading the shipped JSON avoids downloading the datasets on every call.
    Resolve via the top-level ``tnqeet`` package, not ``tnqeet.data`` -- the
    latter's import downloads the datasets.
    """
    resource = resources.files("tnqeet").joinpath("data", "fewshot_examples.json")
    if not resource.is_file():
        raise FileNotFoundError(
            "Bundled fewshot examples not found at tnqeet/data/fewshot_examples.json. "
            "Regenerate them by importing tnqeet.data (or call "
            "tnqeet.data.save_fewshot_examples())."
        )
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


class ArabicDottingSignature(dspy.Signature):
    dotless_text = dspy.InputField(desc="Arabic text without dots (Rasm)")
    dotted_text = dspy.OutputField(desc="Text with proper dots added")


mapping_desc = f"""
    Letter mappings (undotted → possible dotted forms):
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
    - {constants.ALEF_RASM} (ALEF_RASM) → ا أ إ آ""".strip()


class DetailedArabicDotingSignature(dspy.Signature):
    dotless_text = dspy.InputField(
        desc=f"Arabic text without dots (Rasm) - simplified letter forms using direct character mapping as in the following:\n\n{mapping_desc}"
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
            api_base="https://openrouter.ai/api/v1/" if use_openrouter_model else None,
        )
        dspy.configure(lm=self.lm)

        # Prepare few-shot examples if requested
        signature = dspy.Signature({**signature.input_fields, **signature.output_fields}, signature.instructions, signature.__doc__)  # type: ignore
        self.dotter = dspy.Predict(signature=signature)
        if num_fewshot > 0:
            # Default to the bundled fewshot examples (no dataset download).
            if fewshot_dataset is None:
                fewshot_dataset = _load_bundled_fewshot_examples()
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
