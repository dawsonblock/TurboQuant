"""Quick side-by-side test: dense vs TurboQuant KV compression."""
import mlx_lm

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
MAX_TOKENS = 300

# Build a ~10k token prompt by repeating a long document passage
_PASSAGE = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to "
    "the natural intelligence displayed by animals including humans. AI research has been "
    "defined as the field of study of intelligent agents, which refers to any system that "
    "perceives its environment and takes actions that maximize its chance of achieving its "
    "goals. The term 'artificial intelligence' had previously been used to describe machines "
    "that mimic and display human cognitive skills associated with the human mind, such as "
    "learning and problem-solving. This definition has since been rejected by major AI "
    "researchers who now describe AI in terms of rationality and acting rationally, which "
    "does not limit how intelligence can be articulated. AI applications include advanced "
    "web search engines, recommendation systems, understanding human speech, self-driving "
    "cars, generative or creative tools, automated decision-making and competing at the "
    "highest level in strategic game systems. As machines become increasingly capable, "
    "tasks considered to require intelligence are often removed from the definition of AI, "
    "a phenomenon known as the AI effect. Modern machine capabilities generally classified "
    "as AI include successfully understanding human speech, competing at the highest level "
    "in strategic game systems such as chess and Go, and also impersonating humans to the "
    "degree that results are indistinguishable from those produced by humans. AI was founded "
    "as an academic discipline in 1956, and in the years since it has experienced several "
    "waves of optimism, followed by disappointment and the loss of funding, followed by "
    "new approaches, success, and renewed funding. AI research has tried and discarded many "
    "different approaches, including simulating the brain, modeling human problem solving, "
    "formal logic, large databases of knowledge, and imitating animal behavior. In the first "
    "decades of the 21st century, highly mathematical and statistical machine learning has "
    "dominated the field, and this technique has proved highly successful, helping to solve "
    "many challenging problems throughout industry and academia. The various sub-fields of "
    "AI research are centered around particular goals and the use of particular tools. "
    "The traditional goals of AI research include reasoning, knowledge representation, "
    "planning, learning, natural language processing, perception, and the ability to move "
    "and manipulate objects. General intelligence, the ability to complete any task "
    "performable by a human on at least an equal level, is among the field's long-term goals. "
)
# ~170 tokens per passage, repeat ~60x to reach ~10k tokens
PROMPT = (
    "You are a helpful assistant. Here is a long technical document. Read it carefully "
    "and then provide a structured summary with key themes:\n\n"
    + (_PASSAGE * 60)
    + "\n\nNow provide a concise structured summary of the document above."
)

print("Loading model...")
model, tokenizer = mlx_lm.load(MODEL)

# ── Dense baseline ──────────────────────────────────────────────────────────
print("\n=== DENSE (no compression) ===")
dense_response = mlx_lm.generate(
    model, tokenizer,
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
    verbose=True,
)

# ── TurboQuant (wired via generate_step kwargs) ─────────────────────────────
print("\n=== TURBOQUANT (3-bit K + 4-bit V) ===")
tq_response = mlx_lm.generate(
    model, tokenizer,
    prompt=PROMPT,
    max_tokens=MAX_TOKENS,
    verbose=True,
    turboquant_k_start=64,
    turboquant_main_bits=3,
    turboquant_group_size=64,
    turboquant_return_mode="view",
    turboquant_resid_scale_bits=8,
    turboquant_residual_topk=2,
    turboquant_v_bits=4,
    turboquant_v_group_size=64,
    turboquant_v_enabled=True,
    turboquant_rotation="hadamard",
)

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DENSE:      ", dense_response.strip())
print("TURBOQUANT: ", tq_response.strip())
