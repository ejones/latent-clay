import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SkeletonHelper } from 'three';
import { CCDIKSolver } from 'three/addons/animation/CCDIKSolver.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

// Initialize scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(1, 1, 1).normalize();
scene.add(directionalLight);


// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);

const settings = Object.fromEntries(
    ['Left', 'Right'].flatMap(lr =>
        ['Arm', 'ForeArm', 'Hand'].map(name =>
            [`${lr}${name}`, {x: 0, y: 0, z: 0}]
        )
    )
);

const clock = new THREE.Clock();

class BoneTarget {
      constructor(model, bone, effector) {
          this.model = model;
          this.bone = bone;
          this.effector = effector;

          this._positionMap = new Map(
              model.skeleton.bones.map(bone => {
                  const vec = new THREE.Vector3();
                  for (let p = bone; p && p !== this.model; p = p.parent) {
                      vec.add(p.position);
                  }
                  return [bone.name, vec];
              })
          );

          this._vec = new THREE.Vector3();
      }

      setModelRelative(x, y, z) {
          if (y === undefined) {
              this._vec.copy(x); // x is Vector3
          } else {
              if (typeof y === 'string') {
                  for (const [k, v] of this._positionMap) {
                      if (k.endsWith(y)) {
                          y = v.y;
                          break;
                      }
                  }
              }
              this._vec.set(x, y, z);
          }
          for (let p = this.bone.parent; p && p !== this.model; p = p.parent) {
              this._vec.sub(p.position);
          }
          this.bone.position.copy(this._vec);
      }
}

const createPanel = () => {
    const gui = new GUI();
    for (const key in settings) {
        const folder = gui.addFolder(key);
        for (const ax of 'xyz') {
            folder.add(settings[key], ax, -1, 1);
        }
    }
};

const wireUpForIk = (model, specs, rotationConstraints) => {
    const { bones } = model.skeleton;

    const ikPairs = Object.entries(specs).map(([name, spec]) => {
        const effectorBone = bones.find(bone => bone.name.endsWith(spec.effector));
        const linkBones = [];
        const targetPos = new THREE.Vector3();
        targetPos.copy(effectorBone.position);

        let link = effectorBone.parent;
        while (link && !link.name?.endsWith(spec.base)) {
            linkBones.push(link);
            targetPos.add(link.position);
            link.rotation.order = 'YZX'; // rotation constraints depend on this
            link = link.parent;
        }
        if (!link?.name?.endsWith(spec.base)) {
            console.warn('Couldnt find a path from effector to base', linkBones, link);
            return { /* ... ?? */ };
        }

        const baseBone = link;
        const targetBone = new THREE.Bone({name: `target_${effectorBone.name}`});
        targetBone.position.copy(targetPos);
        baseBone.add(targetBone);

        const makeIk = bones => ({
            target: bones.indexOf(targetBone),
            effector: bones.indexOf(effectorBone),
            links: linkBones.map(bone => {
                let constraints;
                for (const key in rotationConstraints) {
                    if (bone.name.endsWith(key)) {
                        constraints = rotationConstraints[key];
                        break;
                    }
                }

                let ikConstraints;
                if (constraints) {
                    const [[xMin, xMax], [yMin, yMax], [zMin, zMax]] = constraints;
                    ikConstraints = {
                        rotationMin: new THREE.Vector3(xMin * Math.PI, yMin * Math.PI, zMin * Math.PI),
                        rotationMax: new THREE.Vector3(xMax * Math.PI, yMax * Math.PI, zMax * Math.PI),
                    };
                } else {
                    console.warn(`Couldn't find rotation constraints for ${bone.name}, setting as rigid`);
                    ikConstraints = {
                        rotationMin: new THREE.Vector3(0, 0, 0),
                        rotationMax: new THREE.Vector3(0, 0, 0),
                    };
                }

                return {
                    index: bones.indexOf(bone),
                    ...ikConstraints,
                };
            }),
            iteration: 10,
            minAngle: 0.0,
            maxAngle: Math.PI
        });

        return [name, { targetBone, makeIk, effectorBone }];
    });
    
    const newBones = [...bones, ...ikPairs.map(([, { targetBone }]) => targetBone)];
    model.bind(new THREE.Skeleton(newBones));

    const targetBones = Object.fromEntries(
        ikPairs.map(([name, { targetBone, effectorBone }]) =>
            [name, new BoneTarget(model, targetBone, effectorBone)]
        )
    );

    const iks = ikPairs.map(([, { makeIk }]) => makeIk(newBones));

    return [targetBones, iks];
};

const updateWave = (targetBone) => {
    const waveAmplitudeX = 20;
    const waveAmplitudeY = 10;
    const waveFrequency = 10;

    const elapsedTime = clock.getElapsedTime();
    const newX = 20 + waveAmplitudeX * Math.cos(waveFrequency * elapsedTime);
    const newY = 40 + waveAmplitudeY * Math.sin(waveFrequency * elapsedTime);

    targetBone.position.set(newX, newY, 0);
};

function animateClap(mgr) {
    const waveFrequency = 18;
    const elapsedTime = clock.getElapsedTime();
    const sine = Math.sin(waveFrequency * elapsedTime)
    const xPos = 10 * (1 + sine) + 2;
    mgr.leftHand.setModelRelative(xPos, 'Spine1', 30);
    mgr.leftHandPerp.setModelRelative(xPos - 8, 'Spine1', 36);
    mgr.rightHand.setModelRelative(-xPos, 'Spine1', 30);
    mgr.rightHandPerp.setModelRelative(-xPos + 8, 'Spine1', 36);
}

/*
TODO adapt to new BoneTarget
function animateHead(targetBones) {
    const headBone = targetBones.head;
    if (headBone) {
        const nodFrequency = 5;
        const elapsedTime = clock.getElapsedTime();
        headBone.position.z = 5 + 5 * Math.sin(nodFrequency * elapsedTime);
    }
}


function animateLegs(targetBones) {
    const leftFootBone = targetBones.leftFoot;
    const rightFootBone = targetBones.rightFoot;
    if (leftFootBone && rightFootBone) {
        const walkFrequency = 10;
        const elapsedTime = clock.getElapsedTime();
        leftFootBone.position.z = 18 * Math.sin(walkFrequency * elapsedTime);
        rightFootBone.position.z = -18 * Math.sin(walkFrequency * elapsedTime);
    }
}
*/

// Load humanoid model
const loader = new GLTFLoader();
loader.load('models/gltf/Xbot.glb', function (gltf) {
    const group = gltf.scene;
    scene.add(group);

    group.position.y -= 1;

    let model;
    group.traverse(node => {
        if (!model && node.isSkinnedMesh) {
            model = node;
        }
    });

    const handPerps = [];
    for (const lr of ['Left', 'Right']) {
        const handBone = model.skeleton.bones.find(({name}) => name.endsWith(`${lr}Hand`));
        const perpBone = new THREE.Bone();
        perpBone.name = `${lr}HandPerp`;
        perpBone.position.set(lr === 'Left' ? 8 : -8, -8, 0);
        handBone.add(perpBone);
        handPerps.push(perpBone);
    }

    model.bind(new THREE.Skeleton([...model.skeleton.bones, ...handPerps]));

    const [targetBones, iks] = wireUpForIk(model, {
        leftHand: {base: 'Shoulder', effector: 'LeftHand'},
        leftHandPerp: {base: 'Shoulder', effector: 'LeftHandPerp'},

        rightHand: {base: 'Shoulder', effector: 'RightHand'},
        rightHandPerp: {base: 'Shoulder', effector: 'RightHandPerp'},

        //head: {base: 'Spine', effector: 'Head'},
        //leftFoot: {base: 'Hips', effector: 'LeftFoot'},
        //rightFoot: {base: 'Hips', effector: 'RightFoot'},
    }, {
        LeftArm: [
            [-0.37, 0.41],
            [-0.65, 0.12],


            [-.45, -.4],
            //[-0.45, 0.45],
        ],
        LeftForeArm: [
            [-0.5, 0.4],
            [-0.84, 0],
            [0, 0],
        ],
        LeftHand: [
            [0, 0],
            [-0.12, 0.12],
            [-0.48, 0.48],
        ],
        RightArm: [
            [-0.37, 0.41],
            [-0.12, 0.65],

            [.4, .45],
            //[-0.45, 0.45],
        ],
        RightForeArm: [
            [-0.5, 0.4],
            [0, 0.84],
            [0, 0],
        ],
        RightHand: [
            [0, 0],
            [-0.12, 0.12],
            [-0.48, 0.48],
        ],
    });

    const skeleton = new SkeletonHelper(group);
    scene.add(skeleton);

    /*
    targetBones.leftHand.effector.rotateX(Math.PI);
    targetBones.rightHand.effector.rotateX(Math.PI);
    */

    const ccdikSolver = new CCDIKSolver(model, iks);

    /*
    model.skeleton.bones.find(({name}) => name.includes('LeftArm')).rotateZ(-0.44 * Math.PI);
    model.skeleton.bones.find(({name}) => name.includes('LeftForeArm')).rotateX(-0.4 * Math.PI);
    model.skeleton.bones.find(({name}) => name.includes('RightArm')).rotateZ(0.44 * Math.PI);
    model.skeleton.bones.find(({name}) => name.includes('RightForeArm')).rotateX(-0.4 * Math.PI);
    */

    function updateTargetBones() {
        //updateWave(targetBones.leftHand);
        //animateHead(targetBones);
        animateClap(targetBones);
        //animateLegs(targetBones);
        ccdikSolver.update();
    }
    
    function updateBonesFromSettings() {
        for (const key in settings) {
            const bone = model.skeleton.bones.find(({name}) => name.endsWith(key));
            if (!bone) {
                console.warn(`No bone for settings key: ${key}!`);
                continue;
            }
            const {x, y, z} = settings[key];
            bone.rotation.set(x * Math.PI, y * Math.PI, z * Math.PI, 'YZX');
        }
    }

    //createPanel();

    // Animation loop
    async function animate() {
        requestAnimationFrame(animate);
        updateTargetBones();
        //updateBonesFromSettings();
        controls.update();
        renderer.render(scene, camera);
        
    }

    animate();
}, undefined, function (error) {
    console.error(error);
});

// Set camera position
camera.position.set(0, 0, 5);
