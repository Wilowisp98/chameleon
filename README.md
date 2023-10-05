# Computer Vision Project

# Chameleon

## Project Overview
This project is undertaken as part of our Computer Vision course, and it is aptly named "Chameleon." The core objective of this project is to develop a system capable of dynamically altering a person's skin color to match that of the object they are touching. For instance, when a person touches a red table, the system should seamlessly transform the color of their skin to match the red hue of the table, all while leaving their clothing unaffected.

## Project Stages
The project unfolds in four distinct stages, each playing a crucial role in achieving the ultimate goal of color transformation:
- Point Detection: To establish a connection between the person's body and the object, we will need to accurately detect the point of contact. (This could be done using techniques like hand or finger tracking)
- Color Detection: Once we have the contact point, we'll need to sample the color of the object accurately. (This can be done by capturing a region around the contact point and analyzing the RGB values of that region, lighting might be a challenge)
- Body and Clothing Segmentation (Second part will be optional): While the second part is optional it can be beneficial for a more realistic and accurate color transformation. We can use semantic segmentation techniques (UNETs as the professor told us) to separate the person's body from their clothing. 
- Color Transformation: We'll need to apply color manipulation techniques, such as color transfer or color mapping, to transform the skin color while preserving texture and detail would be the ideal, however, it can become a pretty difficult task.

## Challenges
Some of the challenges will be:
- Lighting Conditions: Different lighting conditions can affect the perceived color of an object.
- Real-Time Processing: If we aim to achieve real-time color transformation, we'll need to optimize your algorithms for speed.
