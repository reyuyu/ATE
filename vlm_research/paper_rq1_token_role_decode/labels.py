import re
from benchmarks.dataset_registry import digest
from .protocol import SEED, TYPES

FUNCTION = set('a an the of to for from by as than that which who whose it its they their this these those there here and or but nor yet so if because while although when where whether is are was were be been being am has have had does do did can could will would shall should may might must not no also very'.split())
RELATION = set('on in at with over under above below beside behind between near next along across around through against inside outside into onto towards toward within beyond alongside together atop amid among front back facing holding holds held wearing wears worn sitting sits seated standing stands lying lies walking walks running runs riding rides ridden eating eats drinking drinks playing plays looking looks watching watches waiting waits carrying carries carried leaning leans flying flies swimming swims moving moves jumping jumps parked parking driving drives pulling pulls pushing pushes throwing throws catching catches talking talks smiling smiles sleeping sleeps resting rests grazing perched attached mounted covered filled hanging surrounded crossing cooking cutting preparing serving reaching using reading writing working skating skiing snowboarding surfing'.split())
VISUAL = set('person people man men woman women boy boys girl girls child children baby babies dog dogs cat cats bird birds horse horses cow cows sheep elephant elephants bear bears zebra zebras giraffe giraffes animal animals puppy puppies kitten kittens duck ducks chicken chickens motorcycle motorcycles bicycle bicycles bike bikes car cars truck trucks bus buses train trains boat boats airplane airplanes plane planes vehicle vehicles skateboard skateboards snowboard snowboards surfboard surfboards ski skis bench benches chair chairs couch couches sofa sofas bed beds table tables desk desks toilet toilets sink sinks refrigerator refrigerators oven ovens microwave microwaves toaster toasters laptop laptops computer computers keyboard keyboards mouse screen screens television tv monitor monitors phone phones cellphone book books clock clocks vase vases scissors toothbrush umbrella umbrellas handbag handbags tie ties suitcase suitcases backpack backpacks bag bags hat hats helmet helmets cap caps shirt shirts jacket jackets coat coats pants jeans shorts dress dresses skirt skirts shoe shoes boot boots sock socks uniform uniforms glasses sunglasses tie ties suit suits hair face faces head heads hand hands foot feet arm arms leg legs tail tails tree trees grass grassy field fields lawn lawns plant plants flower flowers leaf leaves bush bushes forest sky cloud clouds water sea ocean beach sand sandy snow snowy mountain mountains hill hills rock rocks road roads street streets sidewalk sidewalks path paths bridge bridges building buildings house houses home homes room rooms kitchen bathroom bedroom restaurant shop store city town fence fences wall walls floor floors ceiling door doors window windows counter counters cabinet cabinets shelf shelves shelves sign signs light lights hydrant fire tennis racket rackets ball balls bat bats glove gloves frisbee frisbees kite kites cup cups glass glasses bottle bottles bowl bowls plate plates fork forks knife knives spoon spoons food sandwich sandwiches pizza pizzas cake cakes donut donuts banana bananas apple apples orange oranges broccoli carrot carrots hotdog bread cheese meat fish fruit fruits vegetable vegetables meal meals breakfast dinner lunch pot pots pan pans bowl bowl towel towels blanket blankets pillow pillows box boxes basket baskets container containers basket rope wheel wheels seat seats seat rack tray trays toiletpaper paper photo picture pictures camera cameras toy toys stuffed teddy statue statues flag flags pole poles traffic stop parking meter cone cones boatboard railroad track tracks station airport platform pavement surface dirt ground reflection reflections crowd crowds group groups team player players baseball football basketball soccer sport sports dog leash collar harness red blue green yellow orange purple pink brown black white gray grey silver golden gold colorful bright dark light pale striped spotted wooden metal plastic brick concrete glass stone wool cotton leather large small big little tall short long round rectangular square young old one two three four five six seven eight nine ten several many multiple single few pair couple double triple first second third 1 2 3 4 5 6 7 8 9 10'.split())
# These homographs are not assigned a main semantic role without a POS model.
AMBIGUOUS = set('can will may back front left right orange light glass watch watches fly flies duck ducks tie ties bat bats'.split())
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def classify(word):
    word = word.lower()
    if word in AMBIGUOUS:
        return 'Ambiguous', 'excluded_lexical_homograph_v1'
    if word in RELATION:
        return 'Relation', 'relation_priority_lexicon_v1'
    if word in FUNCTION:
        return 'Function', 'function_lexicon_v1'
    if word in VISUAL:
        return 'Visual', 'visual_candidate_lexicon_v1_not_grounding'
    return 'Unknown', 'not_in_frozen_lexicons'


def align(tokenizer, ids):
    special = set(tokenizer.all_special_ids)
    content = [(i, token) for i, token in enumerate(ids) if token not in special]
    text = tokenizer.decode([t for _, t in content], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    exact = encoded['input_ids'] == [t for _, t in content]
    words = []
    if exact:
        offsets = encoded['offset_mapping']
        for match in WORD.finditer(text):
            indices = [i for i, (a, b) in enumerate(offsets) if max(a, match.start()) < min(b, match.end())]
            role, rule = classify(match.group())
            complete = False
            if len(indices) == 1:
                a, b = offsets[indices[0]]
                complete = text[a:b].strip() == match.group()
            steps = [content[i][0] for i in indices]
            words.append(dict(word=match.group(), char_start=match.start(), char_end=match.end(),
                token_steps=steps, token_ids=[ids[t] for t in steps], token_length=len(indices),
                word_type=role, rule_source=rule, offsets=[offsets[i] for i in indices],
                exact_complete_single_token=complete,
                main_eligible=complete and role in TYPES and len(steps) == 1 and steps[0] >= 1))
    token_rows = []
    for t, token in enumerate(ids):
        matches = [w for w in words if t in w['token_steps']]
        category = 'EOS' if token == tokenizer.eos_token_id else 'Special' if token in special else 'Unaligned' if not exact else 'Punctuation' if not matches else matches[0]['word_type']
        token_rows.append(dict(generation_step=t, token_id=token, token_piece=tokenizer.convert_ids_to_tokens(token),
            decoded_piece=tokenizer.decode([token], clean_up_tokenization_spaces=False), category=category,
            lexical_word=matches[0]['word'] if len(matches) == 1 else None))
    return dict(text=text, roundtrip_exact=exact, words=words, tokens=token_rows)


def select(image_sha, annotation, length):
    anchor = .1 + .8 * int(digest([SEED, 'position-anchor', image_sha])[:12], 16) / 16**12
    chosen = []
    for role in TYPES:
        candidates = [w for w in annotation['words'] if w['main_eligible'] and w['word_type'] == role]
        if not candidates:
            continue
        selected = min(candidates, key=lambda w: (abs(w['token_steps'][0] / max(length - 1, 1) - anchor),
            digest([SEED, 'position-tie', image_sha, role, w['token_steps'][0]])))
        chosen.append(dict(selected, generation_step=selected['token_steps'][0],
            token_id=selected['token_ids'][0], relative_generation_position=selected['token_steps'][0] / max(length - 1, 1), anchor=anchor))
    return chosen
