# Cosine Similarity

The **cosine similarity** measures the similarity between two non-zero vectors of an inner product space. It is defined as:

```math
\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
```

where:
- `A` and `B` are two vectors,
- `A \cdot B` is the dot product of `A` and `B`,
- `\|A\|` and `\|B\|` are the magnitudes (or norms) of `A` and `B`.

## Usage
- Commonly used in text analysis and machine learning to measure document similarity.
- Applied in recommendation systems, clustering, and image recognition.

## Example
If `A = (1, 2, 3)` and `B = (4, 5, 6)`, then:

```math
A \cdot B = (1 \times 4) + (2 \times 5) + (3 \times 6) = 32
```

The magnitudes:

```math
\|A\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{14}
```

```math
\|B\| = \sqrt{4^2 + 5^2 + 6^2} = \sqrt{77}
```

Thus, the cosine similarity is:

```math
\cos(\theta) = \frac{32}{\sqrt{14} \times \sqrt{77}} \approx 0.97
```
