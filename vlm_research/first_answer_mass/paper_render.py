"""Compact full-width case plate; reuses frozen maps and the original renderer."""
import textwrap
from benchmarks.dataset_registry import digest, freeze_json, sha256
from . import analyze
from .finalize import original_result_digest
from .run import OUT, ROOT, read, cohort, verify


def main():
    config=verify()
    records={}
    selected=read(OUT/'showcase_selection.json')['question_ids']
    for sample in cohort():
        if sample['question_id'] in selected:
            record=read(OUT/'checkpoints'/f"{sample['task']}__{sample['question_id']}.json")
            assert record['binding']==digest(config)
            assert record['result_sha256']==original_result_digest(record['result'])
            records[sample['question_id']]=record
    original_save=analyze.save

    def compact_save(fig,stem):
        fig.set_size_inches(7.2,7.6)
        for ax in fig.axes:
            title=ax.get_title()
            if ' | Image mass ' in title:
                ax.set_title(title.replace(' | Image mass ',' | M='),fontsize=8,pad=3)
            for text in ax.texts:
                if ' | QID ' in text.get_text():
                    line=text.get_text().replace('\n',' ')
                    text.set_text('\n'.join(textwrap.wrap(line,91)))
                    text.set_fontsize(8)
        fig.texts[0].set_text('One shared scale per case; M = Image mass; dotted boundary = CLIP crop (outside unobserved).')
        fig.texts[0].set_fontsize(7)
        original_save(fig,stem+'_paper')

    analyze.save=compact_save
    analyze.style()
    analyze.case_figure(records)
    freeze_json(OUT/'paper_render_validation.json',dict(binding=digest(config),
        questions=selected,numeric_changes=False,selection_changes=False,
        layout='7.2in full-width plate; abbreviated mass labels and rewrapped questions only.',
        source_sha256=sha256(ROOT/'first_answer_mass/paper_render.py'),
        artifacts={name:sha256(OUT/name) for name in
            ['figure_y_visual_cases_paper.png','figure_y_visual_cases_paper.pdf']}))


if __name__=='__main__':
    main()
