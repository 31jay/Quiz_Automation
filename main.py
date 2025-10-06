import subprocess
import time
import pytesseract
import cv2
import os
from difflib import SequenceMatcher
from answerProviderMain import AnswerProvider

# Configuration
GUTTER = 0.01
CROP_DIR = 'crop/'
MAX_RETRIES = 3
MAX_QUESTIONS = 1000

# Global state
tap_coords = {}
img_height, img_width = 0, 0
previous_question = ''
retry_count = 0
question_no = ""

os.makedirs(CROP_DIR, exist_ok=True)


# -----------------------
# ADB Utilities
# -----------------------
def adb_screenshot(output_path='screenshot.png'):
    """Capture screenshot from Android device via ADB."""
    try:
        subprocess.run(f'adb exec-out screencap -p > {output_path}', 
                      shell=True, check=True, timeout=10)
        return True
    except:
        print("[ERROR] Screenshot failed")
        return False


def adb_tap_xy(x, y):
    """Perform tap at specified coordinates."""
    try:
        subprocess.run(f"adb shell input tap {int(x)} {int(y)}", 
                      shell=True, check=True, timeout=5)
        return True
    except:
        return False


# -----------------------
# Image Processing & OCR
# -----------------------
def process_image_for_ocr(image_input):
    """Preprocess image for better OCR results."""
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input.copy()
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)
    
    return inverted


def extract_text_data(image_input, lang='nep+eng'):
    """Extract text data using Tesseract OCR."""
    processed = process_image_for_ocr(image_input)
    config = f'--oem 3 --psm 6 -l {lang}'
    ocr_data = pytesseract.image_to_data(processed, config=config, 
                                        output_type=pytesseract.Output.DICT)
    return ocr_data


def detect_submit_button_ocr(ocr_data, min_y=0):
    """Detect Submit button using OCR text detection."""
    submit_keywords = ["Submit", "SUBMIT", "submit", "Done", "DONE", "Confirm", "CONFIRM"]
    
    for i, word in enumerate(ocr_data['text']):
        word_clean = (word or "").strip()
        if not word_clean:
            continue
            
        # Check if word matches submit keywords
        if any(keyword in word_clean for keyword in submit_keywords):
            try:
                y = int(ocr_data['top'][i])
                # Only consider submit buttons below the minimum Y (below options)
                if y > min_y:
                    x = int(ocr_data['left'][i] + ocr_data['width'][i] // 2)
                    y_center = int(ocr_data['top'][i] + ocr_data['height'][i] // 2)
                    print(f"✓ Submit button detected via OCR: '{word_clean}' @ ({x}, {y_center})")
                    return (x, y_center)
            except (ValueError, KeyError):
                continue
    
    return None


def detect_submit_from_screenshot(screenshot_path='screenshot.png', submit_coord_y=0):
    """Take screenshot and detect submit button using OCR."""
    try:
        ocr_data = extract_text_data(screenshot_path, lang='eng')
        submit_coords = detect_submit_button_ocr(ocr_data, min_y=submit_coord_y)
        return submit_coords
    except:
        return None


# -----------------------
# Question Detection
# -----------------------
def detect_question_bounds(ocr_data):
    """Detect question vertical bounds using anchor keywords."""
    global question_no
    
    key_start = ["Question's", "Question", "No:", "No", "/50", "Question's No"]
    start_y = []
    end_y = []
    text_height = 0
    detected_qno = None
    
    for i, word in enumerate(ocr_data['text']):
        word_clean = (word or "").strip()
        if not word_clean:
            continue
        
        try:
            h = int(ocr_data['height'][i])
            if h > text_height:
                text_height = h
        except:
            pass
        
        # Detect question number (e.g., "2/50") - only store, don't print yet
        if '/' in word_clean and any(c.isdigit() for c in word_clean):
            parts = word_clean.split('/')
            if len(parts) == 2 and parts[0].strip().isdigit():
                detected_qno = word_clean
        
        # Detect start markers
        if any(marker in word_clean for marker in key_start):
            try:
                y_pos = int(ocr_data['top'][i] + ocr_data['height'][i])
                start_y.append(y_pos)
            except:
                pass
        
        # Detect question end by '?'
        if '?' in word_clean:
            try:
                y_pos = int(ocr_data['top'][i] + ocr_data['height'][i])
                end_y.append(y_pos)
            except:
                pass
    
    if detected_qno:
        question_no = detected_qno
    
    if not start_y or not end_y:
        return None
    
    return max(start_y), max(end_y), max(text_height, 30)


# -----------------------
# Option Detection (Contour-based)
# -----------------------
def detect_option_contours(image, question_end_y):
    """Detect rectangular option boxes using contour detection."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        
        if y + hh < question_end_y - int(0.01 * h):
            continue
        
        if ww < 0.4 * w or hh < 0.03 * h or hh > 0.15 * h:
            continue
        
        aspect = ww / float(hh) if hh > 0 else 0
        if aspect < 3:
            continue
        
        boxes.append((x, y, ww, hh))
    
    # Relaxed fallback
    if len(boxes) < 2:
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if y + hh < question_end_y - int(0.01 * h):
                continue
            if ww < 0.3 * w or hh < 0.02 * h:
                continue
            aspect = ww / float(hh) if hh > 0 else 0
            if aspect < 2.5:
                continue
            if (x, y, ww, hh) not in boxes:
                boxes.append((x, y, ww, hh))
    
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes


# -----------------------
# Cropping & Text Extraction
# -----------------------
def crop_and_extract_content(y_min, y_max, text_height, image, ocr_data):
    """Crop question and options, extract text, populate tap_coords."""
    global img_height, img_width, tap_coords
    
    # Crop question
    y_min_adj = int(y_min + 3 * GUTTER * img_height)
    y_max_adj = int(y_max + GUTTER * img_height)
    left_pad = int(0.05 * img_width)
    right_pad = int(0.95 * img_width)
    
    question_crop = image[y_min_adj:y_max_adj, left_pad:right_pad]
    if question_crop.size == 0:
        return False
    
    q_path = os.path.join(CROP_DIR, 'question.png')
    cv2.imwrite(q_path, question_crop)
    print(f"✓ Question cropped @ y={y_min_adj}-{y_max_adj}")
    
    q_ocr = extract_text_data(question_crop, lang='nep')
    q_text = ' '.join([t.strip() for t in q_ocr['text'] if t.strip()]).strip()
    
    if not q_text:
        q_ocr = extract_text_data(question_crop, lang='enep+ng')
        q_text = ' '.join([t.strip() for t in q_ocr['text'] if t.strip()]).strip()
    
    tap_coords['question'] = ((left_pad + right_pad) // 2, (y_min_adj + y_max_adj) // 2)
    tap_coords['question_text'] = q_text
    
    # Detect options using contours
    boxes = detect_option_contours(image, y_max_adj)
    
    if boxes and len(boxes) >= 2:
        print(f"✓ Detected {len(boxes)} option contours")
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            margin_x = int(0.02 * img_width)
            margin_y = int(0.01 * img_height)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_width, x + w + margin_x)
            y2 = min(img_height, y + h + margin_y)
            
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            opt_path = os.path.join(CROP_DIR, f"option{idx}.png")
            cv2.imwrite(opt_path, roi)
            
            opt_ocr = extract_text_data(roi, lang='nep')
            opt_text = ' '.join([t.strip() for t in opt_ocr['text'] if t.strip()]).strip()
            
            if not opt_text:
                opt_ocr = extract_text_data(roi, lang='nep+eng')
                opt_text = ' '.join([t.strip() for t in opt_ocr['text'] if t.strip()]).strip()
            
            opt_text = ' '.join(opt_text.split())
            
            if opt_text:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tap_coords[opt_text] = (cx, cy)
                print(f"✓ Option {idx} @ ({cx}, {cy}): {opt_text[:40]}...")
        
        if boxes:
            last_box = boxes[-1]
            last_y = last_box[1] + last_box[3]  # Bottom of last option
            
            # 1. Try to detect Submit button via OCR first
            submit_ocr_coords = detect_submit_button_ocr(ocr_data, min_y=last_y)
            if submit_ocr_coords:
                tap_coords['submit'] = submit_ocr_coords
                tap_coords['submit_detected'] = True  # Flag that submit was found via OCR
                print(f"✓ Submit detected via OCR @ {submit_ocr_coords}")
            else:
                # Fallback to calculated position
                submit_y = int(last_y + 2 * text_height)
                submit_x = img_width // 2
                tap_coords['submit'] = (submit_x, submit_y)
                tap_coords['submit_detected'] = False  # Flag that submit was calculated
                print(f"✓ Submit calculated @ ({submit_x}, {submit_y}) - 2*text_height below last option")
            
            tap_coords['last_option_y'] = last_y
    
    else:
        # Fallback: OCR-based line grouping (from original test.py)
        print("⚠ Contour detection insufficient, using OCR line grouping...")
        
        lines = []
        current_line = []
        prev_top = None
        line_gap = int(0.02 * img_height) + 10
        
        for i, word in enumerate(ocr_data['text']):
            if not word.strip():
                continue
            
            try:
                top = int(ocr_data['top'][i])
                if top < y_max_adj:
                    continue
                
                if prev_top is not None and top - prev_top > line_gap:
                    if current_line:
                        lines.append(current_line)
                        current_line = []
                
                current_line.append(i)
                prev_top = top
            except (ValueError, KeyError):
                continue
        
        if current_line:
            lines.append(current_line)
        
        print(f"✓ Detected {len(lines)} option lines via OCR")
        
        for idx, indices in enumerate(lines[:4], start=1):
            try:
                x1 = min([int(ocr_data['left'][i]) for i in indices])
                y1 = min([int(ocr_data['top'][i]) for i in indices])
                x2 = max([int(ocr_data['left'][i] + ocr_data['width'][i]) 
                         for i in indices])
                y2 = max([int(ocr_data['top'][i] + ocr_data['height'][i]) 
                         for i in indices])
                
                mx = int(0.01 * img_width)
                my = int(0.01 * img_height)
                x1 = max(0, x1 - mx)
                y1 = max(0, y1 - my)
                x2 = min(img_width, x2 + mx)
                y2 = min(img_height, y2 + my)
                
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                opt_path = os.path.join(CROP_DIR, f"option{idx}.png")
                cv2.imwrite(opt_path, roi)
                
                words = [ocr_data['text'][i].strip() for i in indices 
                        if ocr_data['text'][i].strip()]
                opt_text = ' '.join(words)
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tap_coords[opt_text] = (cx, cy)
                print(f"✓ Option {idx} @ ({cx}, {cy}): {opt_text[:40]}...")
                
            except (ValueError, KeyError, IndexError) as e:
                print(f"⚠ Error processing option {idx}: {e}")
                continue
        
        # Submit button for OCR fallback
        if lines:
            last_indices = lines[-1]
            try:
                last_y = max([int(ocr_data['top'][i] + ocr_data['height'][i]) 
                             for i in last_indices])
                
                # 1. Try to detect Submit button via OCR first
                submit_ocr_coords = detect_submit_button_ocr(ocr_data, min_y=last_y)
                if submit_ocr_coords:
                    tap_coords['submit'] = submit_ocr_coords
                    tap_coords['submit_detected'] = True  # Flag that submit was found via OCR
                    print(f"✓ Submit detected via OCR @ {submit_ocr_coords}")
                else:
                    # Fallback to calculated position
                    submit_y = last_y + int(2 * text_height)
                    submit_x = img_width // 2
                    tap_coords['submit'] = (submit_x, submit_y)
                    tap_coords['submit_detected'] = False  # Flag that submit was calculated
                    print(f"✓ Submit calculated @ ({submit_x}, {submit_y}) - 2*text_height below last option")
                
                tap_coords['last_option_y'] = last_y
            except:
                tap_coords['submit'] = (img_width // 2, int(0.85 * img_height))
                print(f"⚠ Submit fallback @ default position")
    
    # Verify we have options
    option_keys = [k for k in tap_coords.keys() 
                  if k not in {'question', 'question_text', 'submit', 'next', 'last_option_y'}]
    
    if not option_keys:
        print("⚠ No options extracted!")
        return False
    
    print(f"✓ Total options extracted: {len(option_keys)}")
    return True


# -----------------------
# Answer Selection (from tess.py logic)
# -----------------------
def get_answer_coordinates():
    """Get answer from AnswerProvider and return coordinates."""
    question = tap_coords.get('question_text', '').strip()
    
    if not question:
        raise RuntimeError("No question text found")
    
    ignore = {'question', 'question_text', 'submit', 'next', 'last_option_y'}
    option_keys = [k for k in tap_coords.keys() if k not in ignore]
    
    if not option_keys:
        raise RuntimeError("No options available")
    
    provider = AnswerProvider(answer_file='answers.json' ,warning_sound='warning.mp3')
    correct_answer = provider.get_answer(question, option_keys)
    
    # Direct match first
    if correct_answer in tap_coords:
        matched_key = correct_answer
    else:
        # Fuzzy matching
        best = None
        best_score = 0.0
        
        for key in option_keys:
            score = SequenceMatcher(None, correct_answer.lower(), key.lower()).ratio()
            if score > best_score:
                best_score = score
                best = key
        
        if best_score < 0.5:
            for key in option_keys:
                if correct_answer.lower() in key.lower() or key.lower() in correct_answer.lower():
                    best = key
                    break
        
        matched_key = best
    
    if not matched_key:
        raise RuntimeError("Answer matching failed")
    
    print(f"✓ Selected answer: {matched_key[:60]}")
    
    answer_coords = tap_coords[matched_key]
    submit_coords = tap_coords.get('submit', (img_width // 2, int(0.85 * img_height)))
    
    return answer_coords, submit_coords


def blind_tap_submit():
    """Blind tap for submit button from bottom UP to just above last option."""
    print("⚠ Performing blind submit tapping from bottom UP to above last option...")
    
    x = img_width // 2
    last_option_y = tap_coords.get('last_option_y', int(0.7 * img_height))
    
    # Tap from bottom UP to just above last option (not equal to it)
    start_y = int(0.95 * img_height)  # Start from near bottom
    end_y = int(last_option_y + 0.03 * img_height)  # Stop 20px below last option to avoid tapping it
    
    # Calculate step size for systematic coverage
    total_range = start_y - end_y
    step_size = max(40, total_range // 8)  # ~8 taps across the range
    
    tap_count = 0
    y = start_y
    while y >= end_y and tap_count < 10:  # Safety limit, going UP
        adb_tap_xy(x, y)
        y -= step_size  # Decrement Y to go UP
        tap_count += 1
        # No sleep for maximum speed
    
    print(f"✓ Performed {tap_count} submit blind taps from y={start_y} UP to y={end_y} (above last option at {last_option_y})")


# -----------------------
# Next Button (tess.py logic)
# -----------------------
def tap_next_button():
    """Take screenshot and tap Next button."""
    if not adb_screenshot('after_submit.png'):
        return False
    
    time.sleep(0.1)
    
    image = cv2.imread('after_submit.png')
    extracted = extract_text_data('after_submit.png', lang='eng')
    
    submit_coord_y = tap_coords.get('submit', (0, 0))[1]
    
    for i, text in enumerate(extracted['text']):
        txt = (text or "").strip()
        if 'Next' in txt:
            try:
                y = int(extracted['top'][i] + extracted['height'][i] // 2)
                x = int(extracted['left'][i] + extracted['width'][i] // 2)
                
                if y > submit_coord_y:
                    print(f"✓ Detected 'Next' button @ ({x}, {y})")
                    adb_tap_xy(x, y)
                    print("✓ Tapped 'Next' button")
                    return True
            except:
                continue
        
        if 'Exit' in txt:
            print("✓ Detected 'Exit' button - Quiz Complete!")
            return False
    
    # 3. Fallback: Fast blind tapping from BOTTOM UP to avoid option taps
    print("⚠ 'Next' button not detected via OCR")
    print("→ Quick blind tapping from bottom UP to avoid options...")
    
    x = img_width // 2
    submit_coord_y = tap_coords.get('submit', (0, 0))[1]
    
    # Tap from bottom of screen UP to submit coordinate (reverse direction)
    start_y = int(0.95 * img_height)  # Start from near bottom
    end_y = max(submit_coord_y, int(0.65 * img_height))  # End at submit or 70% of screen
    
    # Calculate step size for ~8 taps
    step_size = max(40, (start_y - end_y) // 8)
    
    tap_count = 0
    y = start_y
    while y >= end_y and tap_count < 12:  # Safety limit, going UP
        adb_tap_xy(x, y)
        y -= step_size  # Decrement Y to go UP
        tap_count += 1
    
    print(f"✓ Performed {tap_count} quick blind taps from bottom y={start_y} UP to y={end_y}")
    return True


# -----------------------
# Main Loop
# -----------------------
def main():
    global img_height, img_width, tap_coords, previous_question
    global retry_count, question_no
    
    print("=" * 60)
    print("MCQ AUTOMATION STARTED")
    print("=" * 60)
    
    questions_solved = 0
    
    while True:
        tap_coords.clear()
        question_no = ""  # Reset at start of each iteration
        
        # Screenshot
        print("\n" + "─" * 60)
        print("📸 Capturing screenshot...")
        if not adb_screenshot('screenshot.png'):
            print("⚠ Screenshot failed, retrying...")
            time.sleep(1)
            continue
        print("✓ Screenshot captured")
        
        # Load image
        image = cv2.imread('screenshot.png')
        if image is None:
            print("⚠ Failed to load image, retrying...")
            time.sleep(1)
            continue
        
        img_height, img_width = image.shape[:2]
        
        # OCR
        print("🔍 Performing OCR...")
        try:
            ocr_data = extract_text_data(image, lang='nep+eng')
            print("✓ OCR completed")
        except Exception as e:
            print(f"⚠ OCR failed: {e}")
            time.sleep(1)
            continue
        
        # Detect question
        print("🎯 Detecting question bounds...")
        bounds = detect_question_bounds(ocr_data)
        if not bounds:
            print("⚠ Could not detect question bounds")
            time.sleep(1)
            continue
        
        y_min, y_max, text_height = bounds
        print(f"✓ Question bounds detected (y={y_min}-{y_max})")
        if question_no:
            print(f"✓ Detected Question No: {question_no}")
        
        # Extract content
        print("✂️ Cropping question and options...")
        if not crop_and_extract_content(y_min, y_max, text_height, image, ocr_data):
            print("⚠ Content extraction failed")
            time.sleep(1)
            continue
        
        # Check duplicate
        current_question = tap_coords.get('question_text', '')
        
        if current_question == previous_question:
            retry_count += 1
            print(f"⚠ Duplicate question detected (retry {retry_count}/{MAX_RETRIES})")
            if retry_count < MAX_RETRIES:
                time.sleep(0.3)
                continue
            else:
                print("→ Proceeding despite duplicate...")
                retry_count = 0
        else:
            retry_count = 0
            previous_question = current_question
        
        # Display question info
        print(f"\n📋 Question: {current_question[:80]}...")
        
        # Get answer and tap
        try:
            answer_coords, submit_coords = get_answer_coordinates()
            
            print(f"👆 Tapping answer @ ({answer_coords[0]}, {answer_coords[1]})")
            adb_tap_xy(answer_coords[0], answer_coords[1])
            time.sleep(0.1)  # Reduced sleep time
            
            # Handle Submit Button Detection & Tapping (3-step workflow)
            submit_success = False
            
            # Step 1: Check if submit was detected via OCR during question processing
            if tap_coords.get('submit_detected', False):
                print(f"👆 Tapping submit (OCR detected during processing) @ {tap_coords['submit']}")
                adb_tap_xy(tap_coords['submit'][0], tap_coords['submit'][1])
                submit_success = True
            
            # Step 2: If not detected initially, take screenshot and detect submit via OCR
            if not submit_success:
                print("🔍 Submit not OCR-detected initially, taking screenshot for detection...")
                if adb_screenshot('after_answer.png'):
                    time.sleep(0.05)  # Brief pause for screenshot
                    secondary_submit = detect_submit_from_screenshot('after_answer.png', 
                                                                   tap_coords.get('last_option_y', 0))
                    if secondary_submit:
                        tap_coords['submit'] = secondary_submit
                        print(f"👆 Tapping submit (re-detected via OCR) @ {secondary_submit}")
                        adb_tap_xy(secondary_submit[0], secondary_submit[1])
                        submit_success = True
            
            # Step 3: If both OCR attempts failed, use blind tapping from bottom to above last option
            if not submit_success:
                print("⚠ Submit OCR detection failed twice, using blind tapping...")
                blind_tap_submit()
            
            time.sleep(0.1)  # Wait after submit attempt
            
        except Exception as e:
            print(f"⚠ Error during answer selection: {e}")
            # If answer selection fails, try blind submit as last resort
            print("→ Attempting blind submit as fallback...")
            blind_tap_submit()
            time.sleep(0.1)
            continue
        
        # Tap Next
        print("🔄 Looking for Next button...")
        tap_next_button()
        time.sleep(0.2)  # Reduced for faster execution
        
        # Update solved count based on OCR
        questions_solved += 1
        if question_no:
            print(f"\n✅ Solved Question No: {question_no}")
        else:
            print(f"\n✅ Solved Question #{questions_solved}")
        
        # Check if reached max questions
        if question_no:
            try:
                current_num = int(question_no.split('/')[0])
                if current_num >= MAX_QUESTIONS:
                    print(f"\n🎉 Reached maximum questions ({MAX_QUESTIONS})")
                    break
            except:
                pass
    
    print("\n" + "=" * 60)
    print(f"AUTOMATION COMPLETE - {questions_solved} questions answered")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED BY USER]")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")