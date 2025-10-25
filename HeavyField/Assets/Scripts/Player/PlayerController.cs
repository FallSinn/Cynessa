using UnityEngine;
using HeavyField.Systems;
using HeavyField.Weapons;

namespace HeavyField.Player
{
    [RequireComponent(typeof(CharacterController))]
    public class PlayerController : MonoBehaviour
    {
        [Header("Components")]
        [SerializeField] private CharacterController characterController;
        [SerializeField] private Camera playerCamera;
        [SerializeField] private WeaponManager weaponManager;
        [SerializeField] private HUDController hudController;

        [Header("Movement")]
        [SerializeField] private float walkSpeed = 4f;
        [SerializeField] private float sprintSpeed = 7.5f;
        [SerializeField] private float crouchSpeed = 2.5f;
        [SerializeField] private float jumpHeight = 1.25f;
        [SerializeField] private float gravity = -19.62f;
        [SerializeField] private float acceleration = 12f;

        [Header("Crouch")]
        [SerializeField] private float standingHeight = 1.8f;
        [SerializeField] private float crouchingHeight = 1f;
        [SerializeField] private float crouchTransitionSpeed = 8f;

        [Header("Aim")]
        [SerializeField] private float defaultFov = 75f;
        [SerializeField] private float aimFov = 55f;
        [SerializeField] private float aimSmooth = 10f;

        [Header("Look")]
        [SerializeField] private float mouseSensitivity = 2.5f;
        [SerializeField] private float pitchClamp = 80f;

        private Vector3 velocity;
        private Vector3 moveInput;
        private float verticalLook;
        private bool isGrounded;
        private bool isSprinting;
        private bool isCrouched;
        private bool isAiming;

        private void Reset()
        {
            characterController = GetComponent<CharacterController>();
            playerCamera = Camera.main;
            weaponManager = GetComponentInChildren<WeaponManager>();
            hudController = FindObjectOfType<HUDController>();
        }

        private void Awake()
        {
            if (!characterController)
            {
                characterController = GetComponent<CharacterController>();
            }

            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        private void Update()
        {
            if (!playerCamera)
            {
                return;
            }

            HandleLook();
            HandleMovement();
            HandleActions();
            UpdateHUD();
        }

        private void HandleLook()
        {
            float mouseX = Input.GetAxisRaw("Mouse X") * mouseSensitivity;
            float mouseY = Input.GetAxisRaw("Mouse Y") * mouseSensitivity;

            verticalLook -= mouseY;
            verticalLook = Mathf.Clamp(verticalLook, -pitchClamp, pitchClamp);

            playerCamera.transform.localRotation = Quaternion.Euler(verticalLook, 0f, 0f);
            transform.Rotate(Vector3.up * mouseX);
        }

        private void HandleMovement()
        {
            float targetSpeed = walkSpeed;
            isSprinting = Input.GetKey(KeyCode.LeftShift) && !isCrouched && moveInput.magnitude > 0.1f;
            if (isSprinting)
            {
                targetSpeed = sprintSpeed;
            }
            else if (isCrouched)
            {
                targetSpeed = crouchSpeed;
            }

            float inputX = Input.GetAxisRaw("Horizontal");
            float inputZ = Input.GetAxisRaw("Vertical");
            Vector3 forward = transform.forward;
            Vector3 right = transform.right;
            Vector3 desiredMove = (forward * inputZ + right * inputX).normalized * targetSpeed;
            moveInput = Vector3.Lerp(moveInput, desiredMove, acceleration * Time.deltaTime);

            isGrounded = characterController.isGrounded;
            if (isGrounded && velocity.y < 0f)
            {
                velocity.y = -2f;
            }

            if (Input.GetButtonDown("Jump") && isGrounded)
            {
                velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
            }

            if (Input.GetKeyDown(KeyCode.LeftControl))
            {
                ToggleCrouch();
            }

            velocity.y += gravity * Time.deltaTime;

            characterController.Move((moveInput + velocity) * Time.deltaTime);

            float targetHeight = isCrouched ? crouchingHeight : standingHeight;
            characterController.height = Mathf.Lerp(characterController.height, targetHeight, crouchTransitionSpeed * Time.deltaTime);
        }

        private void ToggleCrouch()
        {
            if (isCrouched)
            {
                if (Physics.Raycast(transform.position, Vector3.up, out RaycastHit hit, standingHeight))
                {
                    if (hit.distance < standingHeight - crouchingHeight)
                    {
                        return;
                    }
                }
            }

            isCrouched = !isCrouched;
        }

        private void HandleActions()
        {
            if (!weaponManager)
            {
                return;
            }

            if (Input.GetButton("Fire1"))
            {
                weaponManager.TryFireCurrent();
            }

            if (Input.GetButtonDown("Reload"))
            {
                weaponManager.ReloadCurrent();
            }

            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (scroll > 0.1f)
            {
                weaponManager.NextWeapon();
            }
            else if (scroll < -0.1f)
            {
                weaponManager.PreviousWeapon();
            }

            isAiming = Input.GetMouseButton(1);
            weaponManager.SetAim(isAiming);

            float targetFov = isAiming ? aimFov : defaultFov;
            playerCamera.fieldOfView = Mathf.Lerp(playerCamera.fieldOfView, targetFov, aimSmooth * Time.deltaTime);
        }

        private void UpdateHUD()
        {
            if (!hudController || !weaponManager)
            {
                return;
            }

            hudController.UpdateAmmo(weaponManager.CurrentAmmo, weaponManager.CurrentReserve);
            hudController.UpdateHealth(GetComponent<PlayerHealth>()?.CurrentHealth ?? 0f);
            hudController.UpdateWeaponName(weaponManager.CurrentWeapon?.DisplayName ?? "");
        }
    }
}
