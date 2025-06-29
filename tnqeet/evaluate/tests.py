import jiwer
from tnqeet.evaluate.metrics import wer, cer

def test_wer():
    cases = [
        # ("", ""),
        # ("", "hello world"),
        ("the quick brown fox jumps over the lazy dog near the riverbank", "the quick brown fox jumps over the lazy dog"),
        ("artificial intelligence and machine learning are transforming the way we approach complex problems in various industries", "artificial intelligence and machine learning are transforming the way we approach problems in various industries"),
        ("speech recognition technology has improved dramatically over the past decade thanks to advances in deep learning", "speech recognition technology has improved significantly over the past decade thanks to advances in deep learning"),
        ("natural language processing enables computers to understand analyze and generate human language in meaningful ways", "natural language processing enables computers to understand and generate human language in meaningful ways"),
        ("automated speech recognition systems are now capable of transcribing human speech with remarkable accuracy", "automated speech recognition systems are now capable of transcribing speech with remarkable accuracy"),
        # ("hello", ""),
        ("perfect match", "perfect match"),
        ("completely different sentence", "totally unrelated words here")
    ]
    
    for ref, hyp in cases:
        our_result = wer(ref, hyp)
        jiwer_result = jiwer.wer(ref, hyp)
        status = "✓" if abs(our_result - jiwer_result) < 1e-6 else "✗"
        print(f"{status} WER: {our_result:.3f} vs {jiwer_result:.3f}")

def test_cer():
    cases = [
        # ("", ""),
        # ("", "hello"),
        ("the quick brown fox jumps over the lazy dog and runs through the forest", "the quik brown fox jumps over the lzy dog and runs through the forrest"),
        ("artificial intelligence and machine learning algorithms are revolutionizing data science", "artifical inteligence and machine lerning algorithms are revolutionzing data scince"),
        ("speech recognition technology has become increasingly sophisticated with neural networks", "speach recognition tecnology has become increasngly sophisticated with neural netwrks"),
        ("natural language processing involves computational linguistics and artificial intelligence", "natural languag procesing involves computaional linguistics and artifical inteligence"),
        ("deep learning models require substantial computational resources and large datasets", "dep learning models requir substantial computaional resources and larg datasets"),
        # ("hello", ""),
        ("perfect", "perfect"),
        ("completely", "totally")
    ]
    
    for ref, hyp in cases:
        our_result = cer(ref, hyp)
        jiwer_result = jiwer.cer(ref, hyp)
        status = "✓" if abs(our_result - jiwer_result) < 1e-6 else "✗" # type: ignore
        print(f"{status} CER: {our_result:.3f} vs {jiwer_result:.3f}")

if __name__ == "__main__":
    print("WER Tests:")
    test_wer()
    print("\nCER Tests:")
    test_cer()