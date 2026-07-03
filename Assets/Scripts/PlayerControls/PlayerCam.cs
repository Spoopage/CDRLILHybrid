using Unity.MLAgents;
using UnityEngine;
using Unity.MLAgents.Policies;

public class PlayerCam : MonoBehaviour
{
    public float sensX = 100f;
    public float sensY = 100f;

    public Transform orientation;

    float xRot;
    float yRot;

    [Header("Debug")]
    public bool debugMode = false;

    private BehaviorParameters behaviorParameters;

    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        behaviorParameters = GetComponentInParent<BehaviorParameters>();
    }

    void LateUpdate()
    {
        bool isCommunicatorOn = Academy.Instance.IsCommunicatorOn;
        bool isHeuristicMode = (behaviorParameters != null &&
            behaviorParameters.BehaviorType == BehaviorType.HeuristicOnly);
        bool isInferenceOnly = (behaviorParameters != null &&
            behaviorParameters.BehaviorType == BehaviorType.InferenceOnly);

        // Nonaktifkan mouse kalau: communicator aktif dan bukan heuristic,
        // ATAU kalau inference only tanpa communicator
        if ((isCommunicatorOn && !isHeuristicMode) || isInferenceOnly)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            return;
        }

        if (!Application.isFocused) return;

        if (Cursor.lockState != CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        // Direct mouse input - no smoothing, no dragging
        float mouseX = Input.GetAxis("Mouse X") * sensX * Time.deltaTime;
        float mouseY = Input.GetAxis("Mouse Y") * sensY * Time.deltaTime;

        yRot += mouseX;
        xRot -= mouseY;

        xRot = Mathf.Clamp(xRot, -90f, 90f);

        // Rotate camera and orientation
        transform.rotation = Quaternion.Euler(xRot, yRot, 0);
        if (orientation != null)
            orientation.rotation = Quaternion.Euler(0, yRot, 0);

        if (debugMode)
        {
            // lightweight logging to verify mouse input is present
            if (Mathf.Abs(mouseX) > 0.0001f || Mathf.Abs(mouseY) > 0.0001f)
                Debug.Log($"[PlayerCam] mouseX:{mouseX:F4} mouseY:{mouseY:F4} xRot:{xRot:F2} yRot:{yRot:F2}");
        }
    }

    // Clean up on destroy - reset cursor to normal state
    void OnDestroy()
    {
        // Reset cursor state when scene unloads or object is destroyed
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }

    void OnGUI()
    {
        if (!debugMode) return;
        GUILayout.BeginArea(new Rect(10, 220, 320, 140));
        GUILayout.Label($"xRot: {xRot:F2} yRot: {yRot:F2}");
        GUILayout.Label($"sensX: {sensX} sensY: {sensY}");
        GUILayout.EndArea();
    }
}
