"""J-Lens Interactive Dashboard — Gradio UI entry point.

Usage:
    python ui/app.py  ->  opens http://127.0.0.1:7860
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

from ui.components import (
    generate_and_cache,
    analyze_token_position,
    handle_pause,
    handle_save_snapshot,
    handle_clear,
    handle_freeze,
)

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
.token-box { line-height: 2.2; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; min-height: 60px; }
footer { display: none !important; }
"""


def build_ui():
    with gr.Blocks(title="J-Lens — Interactive Interpretability") as demo:

        gr.Markdown("""
        # J-Lens — Sparse Interpretability Dashboard

        **Generate text** from the model, then **click any token** to see its J-Space representation across all layers.
        """)

        # ── Row 1: Prompt + Generate ─────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="The colour of the planet fourth from the Sun is",
                    lines=2,
                )
                with gr.Row():
                    max_tokens_slider = gr.Slider(5, 80, value=30, step=5, label="Max new tokens")
                    temperature_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                    top_p_slider = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
                generate_btn = gr.Button("Generate & Cache Activations", variant="primary")

            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)

        # ── Row 2: Generated text + token display ─────────────────
        with gr.Row():
            generated_text = gr.Textbox(label="Generated Text", interactive=False, lines=3)

        token_html = gr.HTML(label="Token View (prompt=grey, generated=blue)")

        # ── Row 3: Token selector + analyze ───────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                token_position = gr.Number(
                    label="Token Position to Analyze (0 = first token)",
                    value=0, precision=0, minimum=0,
                )
                analyze_btn = gr.Button("Analyze Selected Token", variant="secondary")

            with gr.Column(scale=1):
                k_slider = gr.Slider(5, 100, value=30, step=5, label="K (top-K tokens)")

        # ── Row 4: Per-layer analysis table ───────────────────────
        layer_table = gr.Markdown("*Generate text, then select a token position and click Analyze.*")

        # ── Row 5: Plot ──────────────────────────────────────────
        fve_plot = gr.Plot(label="FVE per Layer")

        # ── Row 6: Intervention controls ──────────────────────────
        with gr.Accordion("Intervention Controls", open=False):
            with gr.Row():
                alpha_slider = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="alpha (steering)")
                beta_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="beta (merge)")
                lambda_slider = gr.Slider(0.0, 0.2, value=0.05, step=0.005, label="lambda (shrinkage)")

            with gr.Row():
                intervention_mode = gr.Dropdown(
                    choices=["none", "swap", "merge", "steer", "ablate"],
                    value="none", label="Intervention Mode",
                )
                source_token = gr.Textbox(value="Mars", label="Source Token")
                target_token = gr.Textbox(value="Venus", label="Target Token")

            with gr.Row():
                pause_layer = gr.Number(value=18, label="Pause Layer", precision=0)
                merge_layer_a = gr.Number(value=18, label="Merge Layer A", precision=0)
                merge_layer_b = gr.Number(value=20, label="Merge Layer B", precision=0)

            with gr.Row():
                pause_btn = gr.Button("Pause Layer")
                save_snap_btn = gr.Button("Save Snapshot")
                freeze_btn = gr.Button("Freeze All")
                clear_btn = gr.Button("Clear All")

        # ── Events ────────────────────────────────────────────────

        # Generate
        generate_btn.click(
            fn=generate_and_cache,
            inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider],
            outputs=[generated_text, token_html, status_output, token_position],
        )

        # Analyze token
        analyze_btn.click(
            fn=analyze_token_position,
            inputs=[
                token_position, k_slider,
                alpha_slider, beta_slider, lambda_slider,
                source_token, target_token, intervention_mode,
                pause_layer, merge_layer_a, merge_layer_b,
            ],
            outputs=[layer_table, status_output, fve_plot],
        )

        # Also analyze when position changes
        token_position.change(
            fn=analyze_token_position,
            inputs=[
                token_position, k_slider,
                alpha_slider, beta_slider, lambda_slider,
                source_token, target_token, intervention_mode,
                pause_layer, merge_layer_a, merge_layer_b,
            ],
            outputs=[layer_table, status_output, fve_plot],
        )

        # Interventions
        pause_btn.click(fn=handle_pause, inputs=[prompt_input, pause_layer], outputs=status_output)
        freeze_btn.click(fn=handle_freeze, inputs=[], outputs=status_output)
        save_snap_btn.click(fn=handle_save_snapshot, inputs=[], outputs=status_output)
        clear_btn.click(fn=handle_clear, inputs=[], outputs=status_output)

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
