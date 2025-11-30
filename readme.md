Objective 9

1 point 
-- Gerometry : Quadtic --> Cone
-- added a cone.json 
-- added the intersectCone in geometry.py
-- added the parse cone functionality in parser.py
-- you can run this CLI to test : 
    python main.py -i scenes/Cone.json -s

0.5 point
-- Sampling and Recursion : Mirror reflection
-- add the compute_reflection in scene.py
-- you can run this CLI to test : 
    python main.py -i scenes/MirrorBall.json -s


0.5 point
-- Sampling and Recursion : Refraction
-- add the compute_refraction in scene.py
-- you can run this CLI to test : 
    python main.py -i scenes/GlassSphere.json -s


0.5 point
-- Sampling and Recursion : Motion Blur
-- sample with 3 frame for sphere who has motion_dir and count the hit and return total hit/3
-- for shader use the total hit to calculate the fraction of color accumulating 
-- you can run this CLI to test : 
    python main.py -i scenes/Sphere_motion.json -s