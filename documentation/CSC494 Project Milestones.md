# CSC494 Project Milestones

---

### Milestone 1: Literature Review & Dataset Preparation
**Deadline:** June 5th 

**Goals:**
* Complete comprehensive literature review
	*  Include detailed summaries for 3 - 4 papers that are directly relevant to unsupervised occlusion-aware object tracking with GNNs, scene-reconstruction and object segmentation.
	* Briefly summarize 8 - 15 related papers.
	* Discuss proposed idea and how it differs from previous research.
	* Outline hypothesis and architecture in detail.
* Dataset Preparation
	* Download CLEVRAR video training dataset.
	* Organize training and evaluation datasets into separate directories.
	* Write a script to extract relevant information (for object occlusion) for the evaluation dataset.

---

### Milestone 2: Object Detection & Scene Graph Construction
**Deadline:** June 26th

**Goals:**
* Implement Object Detection Model
	* Explore and implement the most effective architecture for keypoint detection and object clustering.
* Scene Graph Construction
	* Implement architecture that uses detected object clusters and spatial data from the object detection model to construct frame-level scene graphs.
	* objects are represented by nodes and edges represent spatial relations.
* Implement an unsupervised independent training loop for evaluation purposes in *Milestone 4*.
* Overall 2 stages of the pipeline should be completed
	* A video is taken as input, and per-frame scene graphs should be constructed.

---

### Milestone 3: GNN & Self-Supervised Learning Framework
**Deadline:** July 17th

**Goals:**
* Design GNN Architecture 
	*  Model uses the constructed scene-graphs for learning spatial relations between detected objects.
	* GNN can predict object position in the next frame, and it should be able to re-detect an object after occlusion occurs.
	* Implement an unsupervised independent training loop for *Milestone 4*.
* Self-Supervised Training Loop
	* Write training framework that allows feedback from the GNN to improve the performance of the object detection model.
	* Determine the most effective loss function to train the GNN and the object detection model. 
* At this point the pipeline should be completed.
	* Input: video -> output: object positions

---

### Milestone 4: Training & Evaluation
**Deadline:** July 31st

**Goals:**
* Train models with the full dataset using appropriate parameters (learning rate, regularization, etc.).
* Evaluate performance related to object detection through occlusions.
* Compare the performance of the pipeline when they use unsupervised training loops independently, and the feedback self-supervised training loop.
---

### Milestone 5: Final Report
**Deadline:** August 14th

**Goals:**
* Write a full report discussing the project in detail
	*  Introduction, discuss the problem, and included the related work from the literature review from *Milestone 1*.
	* Discuss the architecture that was researched throughout the project, and highlight the differences from the previous works (as was done in the literature review).
	* Include the hypothesis, and detail how the results were attained.
	* Discuss the results and the comparisons, and explain them.
	* Discuss limitations, future work, and a conclusion.
* Clean and format code base for clarity and readability.

---
