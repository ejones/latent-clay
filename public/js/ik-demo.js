import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SkeletonHelper } from 'three';
import { CCDIKSolver } from 'three/addons/animation/CCDIKSolver.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GUI } from 'three/addons/libs/lil-gui.module.min.js';

window.THREE = THREE; // DEBUG

// Initialize scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, (window.innerWidth / 2) / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth / 2, window.innerHeight);
document.body.appendChild(renderer.domElement);

const textarea = document.createElement('textarea');
Object.assign(textarea.style, {
  boxSizing: 'border-box',
  background: '#222',
  padding: '30px 20px',
  color: 'white',
  fontFamily: 'monospace',
  fontSize: '18px',
  height: '100vh',
  width: `${window.innerWidth / 2}px`,
  position: 'absolute',
  top: 0,
  right: 0,
  outline: 'none',
  border: '0 none',
});
textarea.value = `left-hand {
  transition: 0.2s all;
  position: head left front at body;
  direction: down slightly right;
}`;
// TODO eye
document.body.appendChild(textarea);

const chatbox = document.createElement('textarea');
Object.assign(chatbox.style, {
  position: 'absolute',
  height: '50px',
  bottom: 0,
  right: '50%',
  left: 0,
  color: 'white',
  background: 'rgba(0 0 0 / 0.4)',
  border: '0 none',
  outline: 'none',
  boxSizing: 'border-box',
  padding: '14px',
  resize: 'none',
  overflow: 'hidden',
  fontSize: '16px',
});
chatbox.value = '> '; // TODO: do as element not text
document.body.appendChild(chatbox);

textarea.onkeydown = e => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp') {
        let {value, selectionStart: start, selectionEnd: end} = e.target;
        while (/[0-9]/.test(value[start - 1])) {
          start -= 1;
        }
        while (/[0-9]/.test(value[end])) {
          end += 1;
        }
        const num = end > start ? Number(value.slice(start, end)) : NaN;
        if (Number.isFinite(num)) {
            e.preventDefault();
            let delta  = e.code === 'ArrowDown' ? -1 : 1;
            if (e.shiftKey) {
              delta *= 10;
            }
            e.target.value = `${value.slice(0, start)}${num + delta}${value.slice(end)}`;
            e.target.selectionStart = start;
            e.target.selectionEnd = end;
        }
    }
};

chatbox.onkeyup = async e => {
    if (e.code === 'Enter') {
        const msg = e.target.value.slice(2); // for "> "
        e.target.value = '> ';
        const response = await fetch('/', {method:'POST', body: msg})
        if (!response.ok) {
            throw new Error('bad response!');
        }
        const aiMsg = document.querySelector('.ai-msg');
        const aiMsgItem = document.createElement('div');
        aiMsgItem.className = 'ai-msg-item';
        aiMsg.appendChild(aiMsgItem);
        let inCode = false;
        const addChunk = (s) => {
            if (inCode) {
                textarea.value += s.replace('{', '{\n  transition: 0.2s all;');
            } else {
                aiMsgItem.textContent += s;
            }
        };
        let chunk = '';
        const processChunk = () => {
            const parts = chunk.split('```');
            chunk = '';
            addChunk(parts[0]);
            if (parts.length > 1) {
                if (!inCode) {
                    textarea.value = '';
                }
                inCode = !inCode;
                addChunk(parts[1]);
            }
        };
        for await (const subChunk of response.body.pipeThrough(new TextDecoderStream())) {
            chunk += subChunk;
            if (!subChunk.includes('`')) {
                processChunk();
            }
        }
        processChunk();
        setTimeout(() => {
            aiMsgItem.className += ' hidden';
            setTimeout(() => {
                aiMsgItem.remove();
            }, 2000);
        }, 2400);
    }
};


const ambientLight = new THREE.AmbientLight(0xa0a0a0);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
directionalLight.position.set(1, 1, 1).normalize();
scene.add(directionalLight);


// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);

const settings = Object.fromEntries(
    ['Left', 'Right'].flatMap(lr => [
        [`${lr}Arm`, {x: 0, y: 0, z: 0}],
        [`${lr}ForeArm`, {x: 0, y: 0, z: 0}],
    ])
);

const clock = new THREE.Clock();

const makePositionMap = (model, skeleton) => {
    const matrixWorldInv = new THREE.Matrix4();
    const boneMatrix = new THREE.Matrix4();
    matrixWorldInv.copy(model.matrixWorld).invert();
    const map = new Map(
        skeleton.bones.flatMap((bone, i) => {
            const vec = new THREE.Vector3();
            boneMatrix.multiplyMatrices(matrixWorldInv, bone.matrixWorld);
            vec.setFromMatrixPosition(boneMatrix);
            //vec.copy(bone.position);
            //bone.parent.localToWorld(vec);
            //bone.getWorldPosition(vec);
            //pmodel.worldToLocal(vec);
            let key = bone.name.toLowerCase();
            key = key.replace(/^mixamorig\d+/, '');
            const entries = [];
            if (key === 'spine2') {
                entries.push(['chest', vec]);
            } else if (key === 'spine1') {
                entries.push(['midtorso', vec]);
            } else if (key === 'spine') {
                entries.push(['stomach', vec]);
            }
            if (key.startsWith('left')) {
                entries.push([key.slice(4), vec]);
            } else if (!key.startsWith('right')) {
                entries.push([key, vec]);
            }
            if (key === 'hips') {
                entries.push(['hip', vec]);
            }
            /*
            if (key === 'lefteye') {
                entries.push(['head', vec]); // REVIEW
            }
            */
            return entries;
        })
    );
    const maxY = Math.max(...[...map.values()].map(({name, y}) => name ? y : -Infinity));
    map.set('abovehead', new THREE.Vector3(0, maxY + 10, 0));
    return map;
};


class BoneTarget {
      constructor(model, bone, links, effector) {
          this.model = model;
          this.bone = bone;
          this.links = links;
          this.effector = effector;

          this._vec = new THREE.Vector3();
          this._vec2 = new THREE.Vector3();
      }
      
      reset() {
          for (const bone of this.links) {
              bone.rotation.set(0, 0, 0);
          }
      }
      
      distanceToEffector({threshold = 1} = {}) {
          this._vec.copy(this.bone.position);
          this.bone.parent.localToWorld(this._vec);
          this._vec2.copy(this.effector.position);
          this.effector.parent.localToWorld(this._vec2);
          return this._vec.distanceTo(this._vec2);
      }

      setModelRelative(x, y, z) {
          if (y === undefined) {
              this._vec.copy(x); // x is Vector3
          } else {
              this._vec.set(x, y, z);
          }
          this.model.localToWorld(this._vec);
          this.bone.parent.worldToLocal(this._vec);
          this.bone.position.copy(this._vec);
      }

      setNormal(x, y, z) {
          // TODO avoid resetting this at the beginning?

          this.effector.rotation.x = 0;
          this.effector.rotation.z = 0;

          this._vec.copy(this.effector.position);
          this.effector.parent.localToWorld(this._vec);
          this.model.worldToLocal(this._vec);

          this._vec2.set(x, y, z);
          this._vec.add(this._vec2);
          this.model.localToWorld(this._vec);
          this.effector.worldToLocal(this._vec);



          this._vec2.copy(this._vec);
          this._vec2.x = 0;
          this._vec2.normalize();


          let vx = 0;
          let {z: vz, y: vy} = this._vec2;

          const [nx, ny, nz] = [0, -1, 0];
          //const rx = Math.acos(nz * vz + ny * vy);
          const rx = Math.atan2( vz * ny - vy * nz, vz * nz + vy * ny );

          this._vec2.set(-1, 0, 0);
          this._vec.applyAxisAngle(this._vec2, rx);

          this._vec2.copy(this._vec);
          this._vec2.z = 0;
          this._vec2.normalize();

          ({x: vx, y: vy} = this._vec2);

          const rz = Math.atan2( nx * vy - ny * vx, vx * nx + vy * ny );

          //console.log(this._vec, rx / Math.PI, rz / Math.PI);
          // TODO: use link bone for one of the rotations as needed, per config
          //console.log(rx / Math.PI, rz / Math.PI);

          this.effector.rotation.x = rx;
          this.effector.rotation.z = rz;
      }
}

const createPanel = () => {
    const gui = new GUI();
    for (const key in settings) {
        const folder = gui.addFolder(key);
        for (const axKey in settings[key]) {
            folder.add(settings[key], axKey, -1, 1);
        }
    }
};

const wireUpForIk = (group, model, yzx, specs, rotationConstraints) => {
    const { bones } = model.skeleton;

    const ikPairs = Object.entries(specs).map(([name, spec]) => {
        const effectorBone = bones.find(bone => bone.name.endsWith(spec.effector));
        const linkBones = [];
        //const targetPos = new THREE.Vector3();
        //targetPos.copy(effectorBone.position);

        let link = effectorBone.parent;
        while (link && !link.name?.endsWith(spec.base)) {
            linkBones.push(link);
            //targetPos.add(link.position);
            link.rotation.order = yzx; // rotation constraints depend on this
            link = link.parent;
        }
        if (!link?.name?.endsWith(spec.base)) {
            console.warn('Couldnt find a path from effector to base', linkBones, link);
            return { /* ... ?? */ };
        }

        const baseBone = link;
        const targetBone = new THREE.Bone({name: `target_${effectorBone.name}`});
        //targetBone.position.copy(targetPos);
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
                    const [xMin, xMax] = constraints[[1, 2, 0][yzx.indexOf('X')]];
                    const [yMin, yMax] = constraints[[1, 2, 0][yzx.indexOf('Y')]];
                    const [zMin, zMax] = constraints[[1, 2, 0][yzx.indexOf('Z')]];

                    console.log({
                        xMin, xMax,
                        yMin, yMax,
                        zMin, zMax,
                    });
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

        return [name, { targetBone, linkBones, makeIk, effectorBone }];
    });
    
    const newBones = [...bones, ...ikPairs.map(([, { targetBone }]) => targetBone)];
    const newBoneInverses = [...model.skeleton.boneInverses, ...ikPairs.map(() => new THREE.Matrix4())];
    model.bind(new THREE.Skeleton(newBones, newBoneInverses), model.bindMatrix);

    const targetBones = Object.fromEntries(
        ikPairs.map(([name, { targetBone, linkBones, effectorBone }]) =>
            [name, new BoneTarget(group, targetBone, linkBones, effectorBone)]
        )
    );

    const iks = ikPairs.map(([, { makeIk }]) => makeIk(newBones));

    return [targetBones, iks];
};

function animateThrow(mgr, elapsedTime) {
    const sine = Math.min(Math.max(0, ((elapsedTime - 0.7) % 2) ** 2 / 0.1), 2) - 1;
    const zPos = 30 * (1 + sine);
    const yPos = 35 * (1 + sine);
    mgr.rightHand.setModelRelative(-30, 84, 0);
    mgr.leftHand.setModelRelative(25, 170 - yPos, zPos);
    mgr.leftHand.setNormal(0, 0.2 - 0.4 * (1 + sine), 1 - 0.8 * (1 + sine));

    //mgr.rightHandPerp.setModelRelative(-19, 84, 0);
    //mgr.leftHandPerp.setModelRelative(20, 170 - 1.2 * yPos, 0.8 * zPos + 10 - 5 * (1 + sine))
}

function animateClap(mgr, elapsedTime) {
    const waveFrequency = 18;
    const sine = Math.sin(waveFrequency * elapsedTime)
    const xPos = 10 * (1 + sine) + 2;
    mgr.leftHand.setModelRelative(xPos, 'Spine1', 30);
    mgr.leftHand.setNormal(-1, 0, 0);
    mgr.rightHand.setModelRelative(-xPos, 'Spine1', 30);
    mgr.rightHand.setNormal(1, 0, 0);
    


    //mgr.leftHand.effector.rotation.z = 0.15 * Math.PI;
    //mgr.leftHand.effector.rotation.x = 0.3 * Math.PI;
    //mgr.leftHandPerp.setModelRelative(xPos - 8, 'Spine1', 36);
    //mgr.rightHand.setNormal(1, 0, 0);
    //mgr.rightHand.effector.rotation.x = 0.5 * Math.PI;
    //mgr.rightHand.effector.rotation.z = -0.1 * Math.PI;
    //mgr.rightHandPerp.setModelRelative(-xPos + 8, 'Spine1', 36);
}

/*
TODO adapt to new BoneTarget
function animateHead(targetBones, elapsedTime) {
    const headBone = targetBones.head;
    if (headBone) {
        const nodFrequency = 5;
        headBone.position.z = 5 + 5 * Math.sin(nodFrequency * elapsedTime);
    }
}


function animateLegs(targetBones, elapsedTime) {
    const leftFootBone = targetBones.leftFoot;
    const rightFootBone = targetBones.rightFoot;
    if (leftFootBone && rightFootBone) {
        const walkFrequency = 10;
        leftFootBone.position.z = 18 * Math.sin(walkFrequency * elapsedTime);
        rightFootBone.position.z = -18 * Math.sin(walkFrequency * elapsedTime);
    }
}
*/

// Load humanoid model
const loader = new GLTFLoader();
loader.load('models/gltf/jennifer.glb', function (gltf) {
    const group = gltf.scene;
    scene.add(group);

    group.scale.set(3.2, 3.2, 3.2);
    group.position.y -= 3.2;

    let model;
    group.traverse(node => {
        // TODO more checks to pick right one if multiple skinned meshes in object
        if (!model && node.isSkinnedMesh) {
            model = node;
        }
    });

    //console.log(...model.skeleton.bones.map(({name}) => name));

    /*
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
    */


    // jennifer:
    //  X: up/down
    //  Y: twist
    //  Z: front/back
    // ZXY

    
    

    // xbot:
    //  X: twist
    //  Y: front/back
    //  Z: up/down
    // YZX

    const modelYZX = 'YZX'; // TODO derive from model (or normalize model)
    const [targetBones, iks] = wireUpForIk(group, model, modelYZX, {
        leftHand: {base: 'Shoulder', effector: 'LeftHand'},
        //leftHandPerp: {base: 'Shoulder', effector: 'LeftHandPerp'},

        rightHand: {base: 'Shoulder', effector: 'RightHand'},
        //rightHandPerp: {base: 'Shoulder', effector: 'RightHandPerp'},

        //head: {base: 'Spine', effector: 'Head'},
        //leftFoot: {base: 'Hips', effector: 'LeftFoot'},
        //rightFoot: {base: 'Hips', effector: 'RightFoot'},
    }, {
        LeftArm: [
            [-0.37, 0.41],
            //[0.4, .41], // throw

            [-0.65, 0.12],


            //[-.45, -.4], // "arms at sides"
            [-0.45, 0.45],
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

            //[.4, .45], // "arms at sides"
            [-0.45, 0.45],
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

    function updateTargetBones(elapsedTime) {
        //animateHead(targetBones, elapsedTime);
        //animateThrow(targetBones, elapsedTime);
        animateClap(targetBones, elapsedTime);
        //animateLegs(targetBones, elapsedTime);
        ccdikSolver.update();
    }
    */
    
    function updateBonesFromSettings() {
        for (const key in settings) {
            const bone = model.skeleton.bones.find(({name}) => name.endsWith(key));
            if (!bone) {
                console.warn(`No bone for settings key: ${key}!`);
                continue;
            }

            /*
            const rot = bone.rotation;
            const {x = rot.x / Math.PI, y = rot.y / Math.PI, z = rot.z / Math.PI} = settings[key];
            bone.rotation.set(x * Math.PI, y * Math.PI, z * Math.PI, 'YZX');
            */

            const {x, y, z} = settings[key];
            if (x || y || z) {
                if (!bone.__initial) {
                    scene.attach(bone);
                    bone.__initial = [...bone.rotation];
                }
                //const par = bone.parent;
                const eu = new THREE.Euler(x * Math.PI, y * Math.PI, z * Math.PI, 'YZX');
                const quat = new THREE.Quaternion().setFromEuler(eu);
                bone.rotation.set(...bone.__initial);
                bone.quaternion.premultiply(quat);
                //par.attach(bone);
            }
        }
    }

    let startTime = 0;
    let targetValues = [0, 0, 0, 0, 0, 0];
    let currentValues = targetValues;
    let startValues = targetValues;

    let positionMap, boneKeywords;

    const xzVec = new THREE.Vector2();

    function updateBonesFromText(elapsedTime) {
        if (elapsedTime < .33) {
          targetBones.rightHand.setModelRelative(0, 0, 0);
          targetBones.leftHand.setModelRelative(0, 0, 0);
          ccdikSolver.update();
          return;
        }

        if (!positionMap) {
            positionMap = makePositionMap(group, model.skeleton);
            boneKeywords = [...positionMap.keys()];

            // DEBUG
            console.log(boneKeywords);
            const partYs = [...positionMap].map(([k, v]) => [k, v.y]).sort(([, a], [, b]) => b - a);
            for (const [k, v] of partYs) {
                console.log(k, v);
            }
            console.log(model);
        }

        const match = textarea.value.match(
          /^\s*left-hand(?:\s*,.*?)?\s*\{\s*transition:[^;]+;\s*position:\s*([a-z_-]+(?:\s+[a-z_-]+)*)\s*;\s*direction:\s*((?:up|left|slightly|right|down|out|in|none)(?:\s+(?:up|left|right|down|out|slightly|in|none))*)/);

        if (match) {
            let [, poss, dirs] = match;
            poss = ` ${poss} `;
            const yKwd = boneKeywords.find(kw => poss.includes(` ${kw} `));
            if (yKwd) {
              const radialDist = (poss.includes('slightly extended')
                ? 40
                : poss.includes('extended')
                ? 80
                : 25) / 100;
              const xDir = 
                poss.includes(' slightly left ') ? 0.5 : poss.includes(' slightly right ') ? -0.5 :
                poss.includes(' left ') ? 1 : poss.includes(' right ') ? -1 : 0;
              const zDir =
                poss.includes(' slightly front ') ? 0.5 : poss.includes(' slightly back ') ? -0.5 :
                poss.includes(' front ') ? 1 : poss.includes(' back ') ? -1 : 0;
              xzVec.set(xDir, zDir).setLength(radialDist);
              
              const newTargetValues = [
                xzVec.x,
                positionMap.get(yKwd).y,
                xzVec.y,

                dirs.includes('slightly left') ? 0.5 : dirs.includes('slightly right') ? -0.5 : dirs.includes('left') ? 1 : dirs.includes('right') ? -1 : 0,
                dirs.includes('slightly up') ? 0.5 : dirs.includes('slightly down') ? -0.5 : dirs.includes('up') ? 1 : dirs.includes('down') ? -1 : 0,
                dirs.includes('slightly out') ? 0.5 : dirs.includes('slightly in') ? -0.5 : dirs.includes('out') ? 1 : dirs.includes('in') ? -1 : 0,
              ];
              const transitionDuration = 0.3;
              const transitionTimingFunction = p => p;
              if (newTargetValues.some((v, i) => v !== targetValues[i])) {
                  //const dbvec = new THREE.Vector3(...newTargetValues.slice(0, 3));
                  //targetBones.leftHand.getParentModel().localToWorld(dbvec);
                  //console.log(newTargetValues.slice(0, 3));
                  //console.log(dbvec.x, dbvec.y, dbvec.z);
                  //console.log(targetBones.leftHand.getParentModel());
                  targetValues = newTargetValues;
                  startValues = currentValues;
                  startTime = elapsedTime;
              }
              const p = transitionTimingFunction(
                  Math.min((elapsedTime - startTime) / transitionDuration, 1)
              );
              currentValues = targetValues.map((tgt, i) => startValues[i] * (1 - p) + tgt * p);
              targetBones.leftHand.setModelRelative(...currentValues.slice(0, 3));
              ccdikSolver.update();
              if (targetBones.leftHand.distanceToEffector() > 1) {
                  targetBones.leftHand.reset();
                  targetBones.leftHand.setModelRelative(...currentValues.slice(0, 3));
                  ccdikSolver.update();
              }
              targetBones.leftHand.setNormal(...currentValues.slice(3));
            }
        }
    }

    createPanel();

    // Animation loop
    const animate = () => {
        requestAnimationFrame(animate);
        const elapsedTime = clock.getElapsedTime();

        //updateBonesFromText(elapsedTime);
        updateBonesFromSettings();

        controls.update();
        renderer.render(scene, camera);
        
    };

    animate();
}, undefined, function (error) {
    console.error(error);
});

// Set camera position
camera.position.set(0, 0, 5);
