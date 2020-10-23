# Dendritic Arborization Tracer

Dendritic Arborization Tracer is a tool to segment neurons along with their dendrites. Ideally, it takes a single channel image as input. 

- The neuron mask is detected in the 'Mask neuron' tab
- The cell body mask is detected in the 'Mask cell body' tab
- Finally, the neuron is skeletonized in the 'Segment dendrites' tab using skeletonization or using the watershed algotrithm. Once skeletonized, the neuron can be further segmented to identify dendrites.

# Install

1. Install [python 3.7](https://www.python.org/downloads/) or [Anaconda 3.7](https://www.anaconda.com/distribution/) (if not already present on your system)

2. In a command prompt type: 

    ```
    pip install --user --upgrade dendritic-arborization-tracer
    ```
    or
    ```
    pip3 install --user --upgrade dendritic-arborization-tracer
    ```
    NB:
    - To open a **command prompt** on **Windows** press **'Windows'+R** then type **'cmd'**
    - To open a **command prompt** on **MacOS** press **'Command'+Space** then type in **'Terminal'**

3. To open the graphical user interface, type the following in a command:
    ```
    python -m dendritic_arborization_tracer
    ```
    or
    ```
    python3 -m dendritic_arborization_tracer
    ```