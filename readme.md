Objective 9

1 point 
-- Gerometry : Quadtic --> Cone
-- added a cone.json 
-- added the intersectCone in geometry.py
-- added the parse cone functionality in parser.py
-- you can run this CLI to test : 
    python main.py -i scenes/Cone.json -s
--or
    open "out/Cone.png"

0.5 point
-- Sampling and Recursion : Mirror reflection
-- add the compute_reflection in scene.py
-- you can run this CLI to test : 
    python main.py -i scenes/MirrorBall.json -s
-- or 
    open "out/MirrorBall.png"

0.5 point
-- Sampling and Recursion : Refraction
-- add the compute_refraction in scene.py
-- you can run this CLI to test : 
    python main.py -i scenes/GlassSphere.json -s
-- or 
    open "out/GlassSphere.png"

0.5 point
-- Sampling and Recursion : Motion Blur
-- sample with 3 frame for sphere who has motion_dir and count the hit and return total hit/3
-- for shader use the total hit to calculate the fraction of color accumulating 
-- you can run this CLI to test : 
    python main.py -i scenes/Sphere_motion.json -s
-- or 
    open "out/Sphere_motion.png"  

1.5 point
-- Geometry : Ray marched implicits --> Metaballs
-- use ray marching to determine the surface of a metaball field defined by multiple implicit spheres
-- you can run this CLI to test :
   python main.py -i scenes/Metaball.json -s
-- or 
    open "out/Metaball.png"


2 point
-- Geometry : Bezier surface patches --> Teapot
-- use evalpatch and tesslation to convert
-- following the idea of : https://www.scratchapixel.com/lessons/geometry/bezier-curve-rendering-utah-teapot/bezier-surface.html
-- you can varify the data is in controll points in Teapot.json and run this CLI to test:
    python main.py -i scenes/Teapot.json -s
-- or 
    open "out/Teapot.png"
    