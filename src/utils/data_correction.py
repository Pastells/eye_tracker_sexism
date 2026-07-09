import eyekit

# 1. Define your Text Block (The actual coordinates of the text on screen)
# You need to provide the bounding box (x, y, width, height) of your text,
# as well as the text itself, font, and font size so the algorithm knows the expected lines.
text_block = eyekit.TextBlock(
    text="Si crees que habíamos llegado a los límites...\nestabas muy equivocado...",
    position=(100, 200),  # The (x, y) pixel coordinates of the top-left of the text
    font_face="Consolas",  # Try to match the font in your experiment
    font_size=24,
    line_height=40,
)

# 2. Load your Fixation Data
# Replace this with your actual data from a CSV/Pandas dataframe
# Format: eyekit.FixationSequence([ [x, y, duration], [x, y, duration], ... ])
raw_fixations = eyekit.FixationSequence(
    [
        [120, 230, 150],
        [150, 231, 200],
        [190, 235, 180],
        # ... import all fixations for this trial
    ]
)

# 3. Apply the Drift Correction Algorithm
# Eyekit will map the fixations to the text_block's expected lines.
# You can use "slice", "merge", "stretch", or "warp" (DTW)

corrected_fixations = eyekit.fixation.drift_correct(
    raw_fixations,
    text_block,
    algorithm="slice",  # 'slice' and 'warp' are generally the most accurate for reading
)

# 4. Extract the corrected Y-coordinates
# The algorithm adjusts the Y-coordinates of the fixations so they sit perfectly
# on the center line of the text they were mapped to.
for fixation in corrected_fixations:
    x = fixation.x
    y = fixation.y  # This is now the corrected Y-axis
    duration = fixation.duration

    # You can now save this corrected data back to a CSV or dataframe
