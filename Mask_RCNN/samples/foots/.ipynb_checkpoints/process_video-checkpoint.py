{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/usuario/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "/Users/usuario/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "/Users/usuario/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "/Users/usuario/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "/Users/usuario/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "/Users/usuario/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n",
      "Using TensorFlow backend.\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'utils'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-15-010bdd6ef3e7>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mfoot\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mutils\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mmodellib\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'utils'"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "import sys\n",
    "import foot\n",
    "import utils\n",
    "import model as modellib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'os' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-13-0d7c55bb9454>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mROOT_DIR\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgetcwd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mMODEL_DIR\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mROOT_DIR\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"logs\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mCOCO_MODEL_PATH\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mROOT_DIR\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"mask_rcnn_coco.h5\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexists\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mCOCO_MODEL_PATH\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mutils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdownload_trained_weights\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mCOCO_MODEL_PATH\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'os' is not defined"
     ]
    }
   ],
   "source": [
    "ROOT_DIR = os.path.abspath(\"../../\")\n",
    "MODEL_DIR = os.path.join(ROOT_DIR, \"logs\")\n",
    "COCO_MODEL_PATH = os.path.join(ROOT_DIR, \"mask_rcnn_foot_0028.h5\")\n",
    "if not os.path.exists(COCO_MODEL_PATH):\n",
    "    utils.download_trained_weights(COCO_MODEL_PATH)\n",
    "\n",
    "\n",
    "class InferenceConfig(coco.CocoConfig):\n",
    "    GPU_COUNT = 1\n",
    "    IMAGES_PER_GPU = 1\n",
    "\n",
    "\n",
    "config = InferenceConfig()\n",
    "config.display()\n",
    "\n",
    "model = modellib.MaskRCNN(\n",
    "    mode=\"inference\", model_dir=MODEL_DIR, config=config\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_names = ['BG', 'left_feet', 'right_feet']\n",
    "\n",
    "def random_colors(N):\n",
    "    np.random.seed(1)\n",
    "    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]\n",
    "    return colors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "colors = random_colors(len(class_names))\n",
    "\n",
    "class_dict = {name: color for name, color in zip(class_names, colors)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'BG': (106.34061119915637, 183.68274582775032, 0.029165578422946092),\n",
       " 'left_feet': (77.09480602111914, 37.42275215836383, 23.54634166604344),\n",
       " 'right_feet': (47.49635390130608, 88.11798539597717, 101.17570592882083)}"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def apply_mask(image, mask, color, alpha=0.5):\n",
    "    \"\"\"apply mas to image\"\"\"\n",
    "    for n, c in enumerate(color):\n",
    "        image[:,:,n] = np.where(\n",
    "            mask == 1,\n",
    "            image[:, :, n] * (1 - alpha) + alpha * c,\n",
    "            image[:, :, n]\n",
    "        )\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_instances(image, boxes, masks, ids, names, scores):\n",
    "    \"\"\"\n",
    "        take the image and results and apply the mask, box, and label\n",
    "    \"\"\"\n",
    "    \n",
    "    n_instances = boxes.shape[0]\n",
    "    colors = random_colors(n_instances)\n",
    "    \n",
    "    if not n_instances:\n",
    "        print('No INSTANCES TO DISPLAY')\n",
    "    else:\n",
    "        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]\n",
    "        \n",
    "    for i in range(n_instances):\n",
    "        if not np.any(boxes[i]):\n",
    "            constinue\n",
    "            \n",
    "        y1, x1, y2, x2 = boxes[i]\n",
    "        label = name[ids[i]]\n",
    "        color = class_dict[label]\n",
    "        score = scores[i] if scores is not None else None\n",
    "        caption = '{} {:.2f}'.format(label, score) if score else label\n",
    "        mask = masks[:, :, i]\n",
    "        \n",
    "        image = apply_mask(image, mask, color)\n",
    "        image = cv2.reactangle(image, (x1,y1), (x2,y2), color, 2)\n",
    "        image = cv2.putText(\n",
    "            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2\n",
    "        )\n",
    "\n",
    "    return image\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "unindent does not match any outer indentation level (<tokenize>, line 6)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"<tokenize>\"\u001b[0;36m, line \u001b[0;32m6\u001b[0m\n\u001b[0;31m    capture = cv2.VideoCapture(0)\u001b[0m\n\u001b[0m    ^\u001b[0m\n\u001b[0;31mIndentationError\u001b[0m\u001b[0;31m:\u001b[0m unindent does not match any outer indentation level\n"
     ]
    }
   ],
   "source": [
    "if _name_ == '_main_':\n",
    "     \"\"\"\n",
    "        test everything\n",
    "    \"\"\"\n",
    "\n",
    "    capture = cv2.VideoCapture(0)\n",
    "\n",
    "    # these 2 lines can be removed if you dont have a 1080p camera.\n",
    "    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)\n",
    "    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)\n",
    "\n",
    "    while True:\n",
    "        ret, frame = capture.read()\n",
    "        results = model.detect([frame], verbose=0)\n",
    "        r = results[0]\n",
    "        frame = display_instances(\n",
    "            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']\n",
    "        )\n",
    "        cv2.imshow('frame', frame)\n",
    "        if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "    capture.release()\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
