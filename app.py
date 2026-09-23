"""
Gradio Web Interface for Generative AI Source Citation Assistant.
Designed for execution in Google Colab and local Python environments.
"""

import os
import gradio as gr
from typing import Tuple, List, Dict, Any

from pipeline import CitationAssistantPipeline, PipelineResult
from source_validator import VerificationStatus
from config import config

# Initialize pipeline instance
pipeline = CitationAssistantPipeline()


def get_status_badge(status: VerificationStatus) -> str:
    """Return HTML badge for claim verification status."""
    if status == VerificationStatus.SUPPORTED:
        return '<span style="background-color: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; border: 1px solid #6ee7b7;">🟢 Supported</span>'
    elif status == VerificationStatus.PARTIALLY_SUPPORTED:
        return '<span style="background-color: #fef3c7; color: #92400e; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; border: 1px solid #fcd34d;">🟡 Partially Supported</span>'
    elif status == VerificationStatus.INSUFFICIENT_EVIDENCE:
        return '<span style="background-color: #f3f4f6; color: #4b5563; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; border: 1px solid #d1d5db;">⚪ Insufficient Evidence</span>'
    else:
        return '<span style="background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; border: 1px solid #fca5a5;">🔴 Unsupported</span>'


def run_citation_assistant(
    question: str,
    citation_style: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str]:
    """
    Handler called when user submits a query.
    """
    if not question or not question.strip():
        return (
            "⚠️ Please enter a question or select an example prompt.",
            "N/A",
            "N/A",
            "N/A",
            ""
        )

    def on_progress(msg: str, frac: float):
        progress(frac, desc=msg)

    # Run complete pipeline
    result: PipelineResult = pipeline.run(
        query=question.strip(),
        citation_style=citation_style,
        progress_callback=on_progress
    )

    # 1. Verification Metrics Summary (HTML Cards)
    m = result.metrics
    metrics_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin: 12px 0 20px 0;">
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; font-weight: 600;">Total Claims</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #111827;">{m.total_claims}</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #065f46; text-transform: uppercase; font-weight: 600;">Supported</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #059669;">{m.supported_claims}</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #92400e; text-transform: uppercase; font-weight: 600;">Partially Supp.</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #d97706;">{m.partially_supported_claims}</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #2563eb; text-transform: uppercase; font-weight: 600;">Grounding Rate</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #2563eb;">{m.grounding_rate_pct}%</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; font-weight: 600;">Cited Sources</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #4b5563;">{m.unique_sources_cited}</div>
        </div>
    </div>
    """

    # 2. Grounded Answer Markdown
    grounded_answer_md = f"""### 📝 Grounded Answer with In-Text Citations\n\n{result.citation_result.cited_answer}"""

    # 3. Claims Verification Breakdown (Interactive HTML)
    claims_html_list = []
    for i, val in enumerate(result.validations, start=1):
        badge = get_status_badge(val.status)
        sources_details = ""
        if val.supporting_sources:
            sources_items = []
            for s in val.supporting_sources:
                sources_items.append(
                    f'<li><a href="{s.url}" target="_blank" style="color: #2563eb; text-decoration: underline; font-weight: 500;">{s.title}</a> '
                    f'<span style="color: #6b7280; font-size: 0.85rem;">({s.publisher}, {s.publication_year or "n.d."})</span> '
                    f'<span style="background: #f3f4f6; color: #374151; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; margin-left: 4px;">Authority: {s.authority_score}/100</span>'
                    f'<div style="margin-top: 4px; color: #4b5563; font-size: 0.85rem; font-style: italic; background: #f9fafb; padding: 6px 10px; border-left: 3px solid #cbd5e1;">"{s.snippet[:200]}..."</div>'
                    f'</li>'
                )
            sources_details = f"""
            <div style="margin-top: 8px;">
                <div style="font-weight: 600; font-size: 0.85rem; color: #374151; margin-bottom: 4px;">Cited Evidence Sources:</div>
                <ul style="margin: 0; padding-left: 20px;">
                    {"".join(sources_items)}
                </ul>
            </div>
            """
        else:
            sources_details = '<div style="margin-top: 6px; font-size: 0.85rem; color: #9ca3af; font-style: italic;">No verifying sources found for this claim.</div>'

        claim_card = f"""
        <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <span style="font-weight: 700; color: #1f2937; font-size: 0.95rem;">Claim #{i} <span style="color: #6b7280; font-weight: normal; font-size: 0.85rem;">({val.claim_id})</span></span>
                <div>{badge}</div>
            </div>
            <div style="color: #111827; font-size: 0.95rem; margin-bottom: 8px; font-weight: 500; line-height: 1.4;">
                "{val.claim_text}"
            </div>
            <div style="background: #f8fafc; border-left: 3px solid #64748b; padding: 8px 12px; border-radius: 0 4px 4px 0; font-size: 0.85rem; color: #334155; margin-bottom: 8px;">
                <strong>Entailment Reasoning:</strong> {val.reasoning}
            </div>
            {sources_details}
        </div>
        """
        claims_html_list.append(claim_card)

    claims_breakdown_html = f"""
    <div style="margin-top: 10px;">
        {"".join(claims_html_list)}
    </div>
    """

    # 4. Formatted References Section
    references_md = f"""### 📚 References ({result.citation_style} Style)\n\n"""
    if result.citation_result.references:
        for ref in result.citation_result.references:
            references_md += f"{ref.formatted_text}\n\n"
    else:
        references_md += "*No verifiable citations attached for the claims in this answer.*"

    # 5. BibTeX Export
    bibtex_md = f"```bibtex\n{result.citation_result.bibtex_export}\n```"

    return (
        metrics_html,
        grounded_answer_md,
        claims_breakdown_html,
        references_md,
        bibtex_md,
    )


# Custom CSS for academic polish
custom_css = """
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.gradio-container {
    max-width: 1050px !important;
    margin: 0 auto !important;
}
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #334155;
}
.header-title {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    margin-bottom: 6px;
    color: #f8fafc;
}
.header-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    line-height: 1.5;
}
.card-section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
}
"""


def create_demo() -> gr.Blocks:
    """Construct the Gradio UI Blocks application."""
    with gr.Blocks(title="Generative AI Source Citation Assistant") as demo:
        
        # Header banner
        gr.HTML("""
        <div class="header-box">
            <div class="header-title">🎓 Generative AI Source Citation Assistant</div>
            <div class="header-subtitle">
                An academic framework for grounding LLM-generated responses in verifiable scholarly evidence. 
                Decomposes text into atomic factual claims, searches authoritative sources (.gov, .edu, peer-reviewed journals), 
                evaluates claim-evidence entailment, and generates properly formatted citations.
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Enter your research topic or question",
                    placeholder="e.g. How does artificial intelligence affect modern education and personalized learning?",
                    lines=3,
                    elem_id="query_input"
                )
            with gr.Column(scale=1):
                citation_style_dropdown = gr.Dropdown(
                    label="Citation Format",
                    choices=config.SUPPORTED_CITATION_STYLES,
                    value=config.DEFAULT_CITATION_STYLE,
                    interactive=True,
                    elem_id="style_dropdown"
                )
                submit_btn = gr.Button("🔍 Generate & Verify Citations", variant="primary", elem_id="submit_btn")

        # Example preset queries
        gr.Examples(
            examples=[
                ["How does artificial intelligence affect modern education and personalized learning?", "APA"],
                ["What are the primary drivers of global climate change according to climate scientists?", "IEEE"],
                ["What is quantum computing and what are its key technological milestones?", "MLA"],
            ],
            inputs=[question_input, citation_style_dropdown],
            label="Sample Academic Research Questions"
        )

        # Output Sections
        gr.HTML('<hr style="margin: 20px 0; border: 0; border-top: 1px solid #e2e8f0;">')
        
        metrics_output = gr.HTML(label="Verification Metrics")
        
        with gr.Tabs():
            with gr.TabItem("📝 Grounded Answer & Citations"):
                grounded_answer_output = gr.Markdown()
                references_output = gr.Markdown()
            
            with gr.TabItem("🔍 Claim Verification Breakdown"):
                claims_breakdown_output = gr.HTML()

            with gr.TabItem("📄 BibTeX Export"):
                bibtex_output = gr.Markdown()

        # Connect button click
        submit_btn.click(
            fn=run_citation_assistant,
            inputs=[question_input, citation_style_dropdown],
            outputs=[
                metrics_output,
                grounded_answer_output,
                claims_breakdown_output,
                references_output,
                bibtex_output,
            ]
        )

    return demo


if __name__ == "__main__":
    app = create_demo()
    # share=True enables a public URL suitable for Colab and remote sharing
    is_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
    app.launch(share=is_colab, server_name="0.0.0.0", server_port=7860)
