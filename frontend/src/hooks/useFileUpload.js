/*
 * useFileUpload — drag-and-drop + click-to-browse file upload hook.
 *
 * Returns:
 *   isDragging   — true when a file is being dragged over the drop zone
 *   file         — the currently selected File object (null if none)
 *   previewUrl   — object URL for image preview (null if none selected)
 *   error        — validation error message (null if valid)
 *   getRootProps — spread onto the drop zone container div
 *   getInputProps— spread onto a hidden <input type="file" />
 *   clearFile    — resets all state
 *
 * Accepted formats: TIFF, JPG/JPEG, PNG, DICOM (.dcm)
 * These match the formats listed in the Upload & Analyse design.
 */

import { useState, useRef, useCallback } from 'react'

const ACCEPTED_TYPES = [
  'image/tiff',
  'image/jpeg',
  'image/png',
  'image/jpg',
  'application/dicom',
]

const MAX_SIZE_MB = 50

export function useFileUpload() {
  const [isDragging, setIsDragging]   = useState(false)
  const [file, setFile]               = useState(null)
  const [previewUrl, setPreviewUrl]   = useState(null)
  const [error, setError]             = useState(null)
  const inputRef                      = useRef(null)

  // Validate and accept a file
  const acceptFile = useCallback((incoming) => {
    if (!incoming) return

    const sizeMB = incoming.size / (1024 * 1024)

    // Allow DICOM by file extension since mime type may be generic
    const isDicom = incoming.name.toLowerCase().endsWith('.dcm')
    const validType = ACCEPTED_TYPES.includes(incoming.type) || isDicom

    if (!validType) {
      setError('Unsupported format. Please upload TIFF, JPG, PNG, or DICOM.')
      return
    }
    if (sizeMB > MAX_SIZE_MB) {
      setError(`File too large (${sizeMB.toFixed(1)} MB). Max is ${MAX_SIZE_MB} MB.`)
      return
    }

    setError(null)
    setFile(incoming)
    // Create an object URL for instant image preview
    const url = URL.createObjectURL(incoming)
    setPreviewUrl(url)
  }, [])

  // Clear selected file and revoke the preview URL to free memory
  const clearFile = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setFile(null)
    setPreviewUrl(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }, [previewUrl])

  // ─── Drop zone event handlers ─────────────────────────────────────────
  const onDragEnter = (e) => { e.preventDefault(); setIsDragging(true) }
  const onDragLeave = (e) => { e.preventDefault(); setIsDragging(false) }
  const onDragOver  = (e) => { e.preventDefault() } // required to allow drop

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files[0]
    acceptFile(dropped)
  }, [acceptFile])

  const onClick = () => inputRef.current?.click()

  const onInputChange = (e) => acceptFile(e.target.files[0])

  // Spread these props onto your drop zone container div
  const getRootProps = () => ({
    onDragEnter,
    onDragLeave,
    onDragOver,
    onDrop,
    onClick,
  })

  // Spread these onto a hidden <input type="file" />
  const getInputProps = () => ({
    ref: inputRef,
    type: 'file',
    accept: '.tiff,.tif,.jpg,.jpeg,.png,.dcm',
    onChange: onInputChange,
    className: 'sr-only', // visually hidden, triggered by click
  })

  return { isDragging, file, previewUrl, error, getRootProps, getInputProps, clearFile }
}
