from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "report.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=11,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13.5,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=11.5,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.1,
            leading=10.7,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "")
    )
    return Paragraph(text, style)


def bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, style)) for item in items],
        bulletType="bullet",
        leftIndent=14,
    )


def numbered_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, style)) for item in items],
        bulletType="1",
        leftIndent=14,
    )


def styled_table(rows: list[list[str]], col_widths=None) -> Table:
    st = styles()["table"]
    data = [[p(cell, st) for cell in row] for row in rows]
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f6")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#90a4b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def run_box() -> Table:
    st = styles()
    rows = [
        [p("How to Run", ParagraphStyle(
            "runbox_title",
            parent=st["body"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=12.5,
            spaceAfter=2,
        ))],
        [p("1. Install packages: python -m pip install -r requirements.txt", st["small"])],
        [p("2. Run Part 1: python run.py part1", st["small"])],
        [p("3. Run Part 2 on Kaggle/Colab GPU if possible: python run.py part2 --encoder microsoft/deberta-v3-base", st["small"])],
        [p("4. Optionally compare cached encoders: python run.py compare-encoders", st["small"])],
        [p("5. Run Part 3 after embeddings are cached: python run.py part3 --encoder microsoft/deberta-v3-base", st["small"])],
    ]
    table = Table(rows, colWidths=[16.2 * cm], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8fc")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#7d96ad")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d1dd")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_story():
    st = styles()
    story = []

    story.append(p("Statistical Machine Learning Course Project Report", st["title"]))
    story.append(run_box())
    story.append(Spacer(1, 0.35 * cm))

    story.append(p("Introduction and Dataset used", st["section"]))
    story.append(
        p(
            "This project studies multi-class text classification in a strictly pre-deep learning setting. "
            "The goal is to compare classical machine learning models trained on two different feature representations: "
            "sparse lexical features and frozen contextual embeddings. The learned models remain classical ML models, "
            "while the pretrained language model is used only as a feature extractor, which keeps the work aligned with the project brief.",
            st["body"],
        )
    )
    story.append(
        p(
            "The dataset used is Hugging Face ag_news. A fixed stratified sample of 8,000 documents was created with four classes: "
            "World, Sports, Business, and Sci/Tech. This dataset was chosen because it is valid under the brief, balanced, and suitable "
            "for comparing keyword-heavy sparse methods against contextual embedding methods.",
            st["body"],
        )
    )
    story.append(
        p(
            "From a statistical learning perspective, this project is useful because the same classification problem can be viewed through two different "
            'feature-design philosophies. The first treats documents as sparse token-count vectors, while the second represents them through a fixed embedding '
            "space obtained from a frozen pretrained encoder. By holding the downstream models classical, the comparison isolates the effect of representation.",
            st["body"],
        )
    )

    story.append(p("Dataset Description and EDA", st["section"]))
    story.append(
        styled_table(
            [
                ["Class", "Samples"],
                ["World", "2000"],
                ["Sports", "2000"],
                ["Business", "2000"],
                ["Sci/Tech", "2000"],
            ],
            col_widths=[8 * cm, 4 * cm],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        p(
            "The dataset was split into 6,400 training samples and 1,600 test samples using stratified sampling. "
            "The average text length is 38.03 tokens, the median is 37, the shortest sample has 10 tokens, and the longest has 139 tokens.",
            st["body"],
        )
    )
    story.append(
        p(
            "These summary statistics indicate that AG News consists of short news snippets rather than long-form articles. "
            "That makes the task especially interesting: short documents often contain strong topical keywords, which helps sparse models, "
            "but they also leave less room for context recovery, which is where dense encoder embeddings can help.",
            st["body"],
        )
    )
    story.append(
        bullet_list(
            [
                "Class distribution plot",
                "Text length histogram",
                "Model comparison chart",
                "Confusion matrices for the best sparse and contextual models",
                "Explained variance plot for SVD and PCA",
            ],
            st["body"],
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        p(
            "The class balance is especially useful for evaluation because Accuracy and Macro F1 can be interpreted without worrying about strong label skew. "
            "This means that differences between models are more likely to reflect the quality of feature extraction and decision boundaries rather than imbalance effects.",
            st["body"],
        )
    )
    story.append(
        p(
            "The text length histogram also shows moderate variation rather than extreme outliers. This matters because both sparse vectorizers and fixed-length "
            "transformer encoding benefit from documents that are not excessively long; truncation and rare-token issues become less severe than they would on full news articles.",
            st["body"],
        )
    )

    story.append(p("Methodology", st["section"]))
    story.append(
        p(
            "Two sparse feature spaces were constructed: unigram Bag-of-Words and bigram TF-IDF. "
            "Both pipelines apply lowercasing, stop-word removal, tokenization through the vectorizer, and vocabulary pruning using min_df, max_df, and max_features.",
            st["body"],
        )
    )
    story.append(
        p(
            "For contextual features, an encoder-only transformer is used strictly as a frozen feature extractor. "
            "The implementation now supports several 768-dimensional encoders and defaults to DeBERTa-v3 Base as the strongest recommended candidate. "
            "Two representations are extracted: CLS embeddings and mean-pooled embeddings. Embeddings are computed once and cached in NPZ format.",
            st["body"],
        )
    )
    story.append(
        p(
            "The upgraded workflow separates Part 1 and Part 2 instead of blending them into one run. "
            "Part 1 focuses only on sparse feature engineering and classical ML. Part 2 focuses only on encoder embedding extraction, ideally on Kaggle or Colab GPUs. "
            "A dedicated comparison stage can rank all cached encoders, and Part 3 then consumes the selected embeddings for classical ML training and evaluation.",
            st["body"],
        )
    )
    story.append(
        p(
            "The Bag-of-Words representation captures direct token frequency and is highly interpretable. The bigram TF-IDF representation extends this by encoding "
            "important two-word phrases and down-weighting globally common tokens. This makes TF-IDF better suited for distinguishing categories whose semantics are partly expressed through short phrases.",
            st["body"],
        )
    )
    story.append(
        p(
            "The contextual representation uses the selected encoder only as a feature extractor, not as a trainable classifier. "
            "The CLS vector provides a single summary token representation, while mean pooling averages information across all contextualized token states. "
            "Comparing these two variants helps determine whether global sentence meaning or dedicated classification-token behavior is more useful for this task.",
            st["body"],
        )
    )
    story.append(
        p(
            "Batch inference is used during encoder extraction, and tqdm progress bars are included so long runs are visible and manageable. "
            "This is especially helpful when Part 2 is run on a remote GPU notebook environment.",
            st["body"],
        )
    )
    story.append(
        bullet_list(
            [
                "Logistic Regression",
                "Linear SVM",
                "RBF SVM",
                "KNN",
                "KMeans",
            ],
            st["body"],
        )
    )
    story.append(
        p(
            "KMeans was adapted into a simple classifier by assigning each cluster the majority class from the training split. "
            "The main evaluation metrics are Accuracy and Macro F1-score.",
            st["body"],
        )
    )
    story.append(
        p(
            "Logistic Regression and Linear SVM were chosen because they are strong linear baselines on high-dimensional text features. "
            "RBF SVM was included to test whether non-linear decision boundaries help once the representation is more compact or semantically rich. "
            "KNN serves as a distance-based baseline, while KMeans provides an unsupervised reference point.",
            st["body"],
        )
    )
    story.append(
        p(
            "The implementation also emphasizes efficiency. Sparse features are cached, contextual embeddings are cached, and the final workflow is wrapped in a single launcher so that the project can be rerun without manually reconstructing the pipeline each time.",
            st["body"],
        )
    )

    story.append(p("Experiments", st["section"]))
    story.append(
        numbered_list(
            [
                "Load the ag_news dataset and create a fixed 8,000-sample subset.",
                "Split the data into training and test sets using stratified sampling.",
                "Build Bag-of-Words and bigram TF-IDF sparse features.",
                "Extract CLS and mean-pooled 768-dimensional encoder embeddings and store them in NPZ format.",
                "Train the required classical models on both feature spaces.",
                "Evaluate using Accuracy and Macro F1-score.",
                "Apply Truncated SVD to TF-IDF and PCA to encoder embeddings.",
                "Analyze explained variance and compare reduced-space performance.",
            ],
            st["body"],
        )
    )
    story.append(
        p(
            "For dimensionality reduction, Truncated SVD was used for TF-IDF because it works directly on sparse matrices. "
            "PCA was used for the dense contextual embeddings.",
            st["body"],
        )
    )
    story.append(
        p(
            "The experiment design intentionally compares full and reduced feature spaces. This helps answer two practical questions: "
            "whether compression causes a major drop in classification quality, and whether the richer contextual space is inherently more compact than the sparse lexical space.",
            st["body"],
        )
    )
    story.append(
        p(
            "For the sparse pipeline, a small amount of model selection was used where it provided good trade-offs, while the dense contextual pipeline was kept simpler to improve stability and reproducibility. "
            "This reflects a realistic engineering decision: on small to medium NLP tasks, moderate tuning often gives most of the gain, while heavy search can add cost without proportional benefit.",
            st["body"],
        )
    )
    story.append(
        p(
            "The project outputs figures, tables, and confusion matrices automatically. This makes the workflow suitable not only for a one-time report, but also for repeated comparison if more models or alternative encoders are tested later.",
            st["body"],
        )
    )
    story.append(
        p(
            "To improve creativity in the visual analysis, the project now includes t-SNE projections for sparse reduced features and encoder embeddings. "
            "These visualizations make it easier to judge whether class separation is naturally emerging in the learned representation space.",
            st["body"],
        )
    )

    story.append(p("Results", st["section"]))
    story.append(
        styled_table(
            [
                ["Feature Space", "Best Model", "Accuracy", "Macro F1"],
                ["Unigram Bag-of-Words", "RBF SVM", "0.8706", "0.8700"],
                ["Bigram TF-IDF", "RBF SVM", "0.8831", "0.8830"],
                ["Frozen encoder CLS (baseline run)", "RBF SVM", "0.8844", "0.8843"],
                ["Frozen encoder Mean (baseline run)", "RBF SVM", "0.8969", "0.8969"],
            ],
            col_widths=[5.2 * cm, 4.0 * cm, 3.0 * cm, 3.0 * cm],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        p(
            "In the current local benchmark, the best overall model was a mean-pooled frozen encoder baseline with an RBF SVM, achieving 0.8969 Accuracy and 0.8969 Macro F1. "
            "The best sparse model was bigram TF-IDF with an RBF SVM, achieving 0.8831 Accuracy and 0.8830 Macro F1. "
            "The upgraded codebase is organized so stronger 768-dimensional encoders such as DeBERTa-v3 Base can be extracted separately on Kaggle or Colab GPU and then reused in Part 3.",
            st["body"],
        )
    )
    story.append(
        p(
            "The strongest class in both the sparse and contextual settings was Sports, while Business and Sci/Tech were relatively harder. "
            "This is reasonable because those two classes often share company names, product references, and market-related vocabulary.",
            st["body"],
        )
    )
    story.append(
        styled_table(
            [
                ["Model", "World", "Sports", "Business", "Sci/Tech"],
                ["Best sparse F1", "0.902", "0.949", "0.844", "0.864"],
                ["Best contextual F1", "0.901", "0.966", "0.862", "0.879"],
            ],
            col_widths=[4.0 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        styled_table(
            [
                ["Reduced Feature Space", "Best Model", "Accuracy", "Macro F1"],
                ["TF-IDF + SVD (300)", "RBF SVM", "0.8831", "0.8829"],
                ["Embeddings + PCA (100)", "RBF SVM", "0.8919", "0.8917"],
            ],
            col_widths=[5.4 * cm, 4.0 * cm, 3.0 * cm, 3.0 * cm],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        p(
            "The explained variance results show that TF-IDF with 300 SVD components retained 0.2659 cumulative explained variance, "
            "while the 100-component PCA representation of mean-pooled embeddings retained about 0.8510.",
            st["body"],
        )
    )
    story.append(
        p(
            "This is a useful contrast. Even though the sparse TF-IDF system remains competitive, its information is spread across a very large basis. "
            "The contextual embedding space, in contrast, keeps much more of its predictive structure even after a stronger dimensionality reduction step.",
            st["body"],
        )
    )
    story.append(
        p(
            "Another important observation is that linear methods remained surprisingly strong in both sparse and contextual settings. "
            "The best contextual model was non-linear, but Logistic Regression and Linear SVM stayed close behind, especially on the mean-pooled embedding representation.",
            st["body"],
        )
    )

    story.append(p("Project Files and Implementation Roles", st["section"]))
    story.append(
        p(
            "For completeness and reproducibility, the project folder contains a small set of top-level files and a code directory. "
            "Each file has a specific role in the workflow rather than being included as redundant support material.",
            st["body"],
        )
    )
    story.append(
        bullet_list(
            [
                "report.pdf: the final written report submitted for the course project.",
                "requirements.txt: the Python package list needed to run the project on a new machine.",
                "run.py: the executable entry point with argparse-style stage selection for Part 1, Part 2, Part 3, and encoder comparison.",
            ],
            st["body"],
        )
    )
    story.append(
        p(
            "The code directory contains the main implementation files. Their roles are summarized below.",
            st["body"],
        )
    )
    story.append(
        bullet_list(
            [
                "config.py: stores project-wide constants such as dataset size, random seed, embedding model name, and directory paths.",
                "data.py: downloads, samples, labels, and splits the AG News dataset into training and test sets.",
                "encoder_registry.py: lists the supported 768-dimensional encoder-only models and the recommended default.",
                "sparse_features.py: builds the unigram Bag-of-Words and bigram TF-IDF pipelines and caches the sparse matrices.",
                "embeddings.py: extracts frozen encoder CLS and mean-pooled embeddings, performs batched inference, validates cached arrays, and stores them in NPZ format.",
                "models.py: defines the classical learning models, dimensionality reduction utilities, and limited tuning logic.",
                "evaluation.py: computes Accuracy, Macro F1, confusion matrices, and structured result tables.",
                "plots.py: generates the visual outputs such as class distribution, comparison charts, confusion heatmaps, and t-SNE graphs.",
                "part1_sparse.py: runs only the sparse-feature stage required for Part 1.",
                "part2_embeddings.py: runs only the encoder embedding extraction stage required for Part 2, ideally on Kaggle or Colab GPU.",
                "part3_contextual.py: trains classical ML models using the cached encoder embeddings from Part 2.",
                "encoder_comparison.py: evaluates all cached encoders and writes a leaderboard to help select the most effective one.",
                "cli.py: stage dispatcher used by the executable file.",
                "run_experiments.py: shared orchestration helpers reused by the modular parts.",
                "generate_clean_report_pdf.py: rebuilds the final PDF report in a clean layout.",
            ],
            st["body"],
        )
    )
    story.append(
        p(
            "This separation makes the project easier to understand. Data handling, feature engineering, model training, evaluation, visualization, and reporting are all modularized, "
            "which improves both readability and maintainability.",
            st["body"],
        )
    )

    story.append(p("Discussion", st["section"]))
    story.append(
        p(
            "The results confirm that sparse lexical features remain strong for short news classification, especially when using TF-IDF with bigrams. "
            "However, the contextual embedding representation provided the best final performance, showing that semantic information still offers a measurable advantage.",
            st["body"],
        )
    )
    story.append(
        p(
            "Among the contextual variants, mean pooling performed better than CLS pooling. This suggests that for short text classification, "
            "aggregating token information across the whole sequence produces a more robust representation than relying only on the CLS token.",
            st["body"],
        )
    )
    story.append(
        p(
            "Dimensionality reduction preserved a large portion of embedding performance, which indicates that the dense semantic representation is relatively compact. "
            "By contrast, sparse TF-IDF needed a higher-dimensional basis to stay competitive.",
            st["body"],
        )
    )
    story.append(
        p(
            "The reduced embedding model remained very close to the best unreduced contextual system, which suggests that most of the task-relevant signal was concentrated "
            "in a smaller subspace. This makes the embedding representation attractive not only for accuracy, but also for compression.",
            st["body"],
        )
    )
    story.append(
        p(
            "From the standpoint of classical machine learning, the most interesting lesson is that the downstream classifier family is not the whole story. "
            "Feature quality strongly shapes the apparent usefulness of linear versus non-linear models. A weak feature space can make a model look underpowered, "
            "while a strong representation can make even simple classifiers highly competitive.",
            st["body"],
        )
    )
    story.append(
        p(
            "The weaker KMeans and KNN results also reinforce an important point. Not every classical method is equally well suited to high-dimensional NLP data. "
            "Distance-based and clustering methods can still be informative as baselines, but supervised discriminative methods remain better choices when labels are available.",
            st["body"],
        )
    )
    story.append(
        p(
            "There is also a practical reproducibility lesson. A full project should not only maximize performance, but also be easy to rerun. "
            "That is why this implementation includes caching and a single launcher script. In real ML workflows, reproducibility and usability are often as important as a small gain in validation score.",
            st["body"],
        )
    )

    story.append(p("Conclusion and Learnings", st["section"]))
    story.append(
        p(
            "This project shows that strong NLP classification performance can be achieved using classical machine learning when the feature representation is carefully designed. "
            "Bigram TF-IDF provides a strong, interpretable baseline, while frozen encoder embeddings offer the best overall performance without violating the project constraint against fine-tuning.",
            st["body"],
        )
    )
    story.append(
        p(
            "The main learning is that feature representation matters at least as much as classifier choice. "
            "Sparse methods are efficient and transparent, whereas contextual embeddings are more expressive and compact under dimensionality reduction.",
            st["body"],
        )
    )
    story.append(
        p(
            "Overall, the project supports a balanced conclusion. Classical sparse NLP techniques are still very effective for topic classification, especially on short text. "
            "At the same time, frozen contextual embeddings offer a practical bridge to modern representation learning and can improve performance even when no deep classifier is trained.",
            st["body"],
        )
    )
    story.append(
        p(
            "If this work were extended further, the most natural next steps would be to benchmark all supported 768-dimensional encoders under the same frozen setup, evaluate a second Hugging Face dataset, "
            "and perform a deeper class-wise error analysis to understand where contextual semantics help the most.",
            st["body"],
        )
    )

    return story


def main():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.85 * cm,
        rightMargin=1.85 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )
    doc.build(build_story(), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
