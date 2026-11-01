"""The user's algorithm: class mask -> greedy S* -> contextual LOO details.

Encoder contract: encode(list[str]) returns unit-normalized Torch features.
All image/class inputs are also normalized. No review lists, LLM calls,
handwritten captions or image-specific exceptions are accepted.
"""
from dataclasses import asdict, dataclass
import re


STOP = frozenset('a an the and or but of in on at for to from by with without as is are was were be been being this that these those it its i me my we our you your he she his her they their s t photo photograph picture image photographs pictures images seen showing show view'.split())
TEMPLATE = frozenset('stock royalty copyright website http https www file download sale buy product premium quality collection edition author person'.split())
ALIASES = {
    'person': ['person','persons','people','man','men','woman','women','boy','boys','girl','girls','child','children'],
    'tv': ['tv','television','televisions'], 'sofa': ['sofa','sofas','couch','couches'],
    'car': ['car','cars','automobile','automobiles'],
    'dog': ['dog','dogs','puppy','puppies'], 'cat': ['cat','cats','kitten','kittens'],
    'horse': ['horse','horses'], 'bear': ['bear','bears'],
    'bird': ['bird','birds'], 'bottle': ['bottle','bottles'], 'orange': ['orange','oranges'],
}


@dataclass(frozen=True)
class Config:
    eta: int = 2
    detail_k: int = 3
    grounding_threshold: float = 0.15
    detail_coefficient: float = 0.1
    class_metric: str = 'margin'
    probability_temperature: float = 0.07
    detail_min_contribution: float = 0.0
    max_detail_words: int = 3

    def __post_init__(self):
        if self.eta < 0 or self.detail_k < 0 or self.max_detail_words < 0:
            raise ValueError('Search and detail budgets must be nonnegative')
        if not -1 <= self.grounding_threshold <= 1 or self.detail_coefficient < 0:
            raise ValueError('Invalid grounding threshold or coefficient')
        if self.class_metric not in ('margin','probability') or self.probability_temperature <= 0:
            raise ValueError('Invalid class-discrimination metric')


def identities(category):
    word = category.lower().strip()
    values = set(ALIASES.get(word, [word, word+'s']))
    if word.endswith('y'):
        values.add(word[:-1]+'ies')
    return values


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def mask_class(text, category):
    # Remove placeholders before tokenization; class words cannot enter D.
    text = re.sub(r'<[^>]*>', ' ', text)
    blocked = identities(category)
    return [w for w in words(text) if w not in blocked]


def coefficient_for_round(start, end, round_index, rounds):
    """Offline text-pool round; never victim-training epoch/native beam round."""
    if rounds < 1 or not 0 <= round_index < rounds or start < 0 or end < start:
        raise ValueError('Invalid offline curriculum')
    return start if rounds == 1 else start+(end-start)*round_index/(rounds-1)


class ContextualDetails:
    def __init__(self, encode, centers, labels, config=None):
        import torch
        self.torch = torch
        self.encode, self.labels = encode, list(labels)
        self.config = config or Config()
        self.centers = torch.nn.functional.normalize(centers.float(), dim=-1)
        if len(self.labels) < 2 or self.centers.shape[0] != len(self.labels):
            raise ValueError('Need aligned centers for all classes')

    def _scores(self, sentences, image, category):
        image = self.torch.nn.functional.normalize(image.float().reshape(-1), dim=0)
        feature = self.encode(sentences)
        image_cos = feature @ image
        classes = feature @ self.centers.T
        target = self.labels.index(category)
        target_cos = classes[:, target]
        others = self.torch.cat([classes[:,:target], classes[:,target+1:]], dim=-1)
        margin = target_cos-others.max(dim=-1).values
        probability = (classes/self.config.probability_temperature).softmax(dim=-1)[:,target]
        return dict(image=image_cos, target=target_cos, margin=margin, probability=probability,
                    # Keep the paper's image-minus-class background objective.
                    objective=image_cos-target_cos)

    def analyze(self, caption, category, image):
        tokens = mask_class(caption, category)
        masked = ' '.join(tokens)
        deletions = []
        with self.torch.inference_mode():
            for step in range(min(self.config.eta, max(0,len(tokens)-1))):
                variants = [' '.join(tokens)] + [' '.join(tokens[:i]+tokens[i+1:]) for i in range(len(tokens))]
                score = self._scores(variants, image, category)
                # Option0 is no deletion; ties keep the current sentence.
                best = max(range(len(variants)), key=lambda j:(float(score['objective'][j]), -j))
                if best == 0:
                    break
                index = best-1
                deletions.append(dict(step=step, word=tokens[index], position=index,
                    objective_gain=float(score['objective'][best]-score['objective'][0])))
                tokens.pop(index)
            background = ' '.join(tokens)
            variants = [background]+[' '.join(tokens[:i]+tokens[i+1:]) for i in range(len(tokens))]
            score = self._scores(variants, image, category)
            contributions = []
            for i, token in enumerate(tokens):
                delta_image = float(score['image'][0]-score['image'][i+1])
                delta_class = float(score[self.config.class_metric][0]-score[self.config.class_metric][i+1])
                eligible = (token not in STOP and token not in TEMPLATE and len(token)>1
                    and token not in identities(category))
                contributions.append(dict(word=token, position=i, delta_image=delta_image,
                    delta_class=delta_class, detail_score=delta_image-delta_class,
                    eligible=(eligible and delta_image>0
                        and delta_image-delta_class > self.config.detail_min_contribution)))
            ranked = sorted((x for x in contributions if x['eligible']), key=lambda x:(-x['detail_score'],x['position']))
            selected, seen = [], set()
            for item in ranked:
                if item['word'] in seen:
                    continue
                if len(selected) >= min(self.config.detail_k,self.config.max_detail_words):
                    break
                selected.append(item); seen.add(item['word'])
            # Original order is used; no LLM rewrites D into a sentence.
            detail_words = [x['word'] for x in sorted(selected,key=lambda x:x['position'])]
        assert not set(detail_words).intersection(identities(category))
        image_cos, class_cos = float(score['image'][0]), float(score['target'][0])
        return dict(caption=caption, category=category, masked_text=masked, background=background,
            greedy_deletions=deletions, contributions=contributions, details=detail_words,
            detail_text=' '.join(detail_words), image_grounding=image_cos,
            background_class_cos=class_cos,
            grounding_pass=bool(tokens) and image_cos >= self.config.grounding_threshold,
            full_class_identity_masked=True, config=asdict(self.config),
            interpretation='CLIP grounding proxy, not independently certified visual truth')


def choose_slots(anchors, candidates, per_class=5):
    """Filter by grounding, rank only by background/class cosine, fill all slots.

    Anchors are the author's original100 with their computed analysis. Existing
    captions participate under the same gates. A shortage retains originals and
    is explicit; no class is assigned a maximum-one replacement rule.
    """
    from collections import Counter
    normalize = lambda s:' '.join(s.lower().split())
    classes = sorted({x['category'] for x in anchors})
    if any(sum(x['category']==c for x in anchors)!=per_class for c in classes):
        raise ValueError('Anchor class quota differs from protocol')
    result = [dict(x) for x in anchors]
    decisions = []
    for category in classes:
        slots = [i for i,x in enumerate(anchors) if x['category']==category]
        existing = {normalize(anchors[i]['caption']):i for i in slots}
        unique = {}
        # Anchors first: exact score ties preserve an existing caption.
        for order,item in enumerate([*(anchors[i] for i in slots),*(x for x in candidates if x['category']==category)]):
            if not item['analysis']['grounding_pass']:
                continue
            key = normalize(item['caption'])
            if key not in unique:
                unique[key] = (item, order)
        ranked = sorted(unique.values(), key=lambda x:(-x[0]['analysis']['background_class_cos'],x[1]))
        top = [x[0] for x in ranked[:per_class]]
        selected_names = {normalize(x['caption']) for x in top}
        retained = {i for name,i in existing.items() if name in selected_names}
        fresh = [x for x in top if normalize(x['caption']) not in existing]
        free = sorted((i for i in slots if i not in retained),
            key=lambda i:(anchors[i]['analysis']['grounding_pass'],anchors[i]['analysis']['background_class_cos'],i))
        for i,new in zip(free,fresh):
            result[i] = dict(new)
            decisions.append(dict(old_index=i,category=category,old=anchors[i],new=new,
                reason='Grounding filter followed by unchanged masked-background/class ranking'))
    assert Counter(x['category'] for x in result)==Counter(x['category'] for x in anchors)
    assert len({x['old_index'] for x in decisions})==len(decisions)
    return result, decisions
