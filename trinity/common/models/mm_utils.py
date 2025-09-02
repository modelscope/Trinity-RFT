from typing import Any, Dict


def build_multi_modal_inputs(
    prompt: str,
    raw_mm_data: Dict[str, Any],
    processor: Any,
    **kwargs,
) -> Dict[str, Any]:
    """
    Preprocess multi-modal data and build multi-modal inputs
    Adapted from: verl/utils/dataset/rl_dataset.py
    """
    from verl.utils.dataset.vision_utils import process_image, process_video

    if prompt is None:
        raise ValueError("Prompt is required for build multi-modal inputs")

    raw_images, raw_videos = None, None
    if "image" in raw_mm_data:
        raw_images = raw_mm_data["image"]
    if "video" in raw_mm_data:
        raw_videos = raw_mm_data["video"]

    multi_modal_data = {}
    images, videos = None, None
    if raw_images is not None:
        images = [process_image(image) for image in raw_images]
        multi_modal_data["image"] = images
    if raw_videos is not None:
        videos = [process_video(video) for video in raw_videos]
        multi_modal_data["video"] = [video.numpy() for video in videos]

    model_inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")

    model_inputs.pop("input_ids", None)  # TODO: check
    model_inputs.pop("attention_mask", None)

    # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
    multi_modal_inputs = dict(model_inputs)

    return {
        "prompt": prompt,
        "multi_modal_inputs": multi_modal_inputs,
        "multi_modal_data": multi_modal_data,
    }


def attach_images_to_messages(messages, raw_mm_data):
    new_msgs = [dict(m) for m in messages]
    imgs = (raw_mm_data or {}).get("image") or []
    if not imgs:
        return new_msgs

    for i in range(len(new_msgs) - 1, -1, -1):
        if new_msgs[i].get("role") == "user":
            content = new_msgs[i].get("content", "")
            items = []
            if isinstance(content, str):
                text = content.replace("<image>", "").replace("<|image_pad|>", "").strip()
                if text:
                    items.append({"type": "text", "text": text})
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, str):
                        t = c.replace("<image>", "").replace("<|image_pad|>", "").strip()
                        if t:
                            items.append({"type": "text", "text": t})
                    elif isinstance(c, dict):
                        items.append(c)

            for img in imgs:
                items.append({"type": "image", "image": img})

            new_msgs[i]["content"] = items
            break

    return new_msgs
